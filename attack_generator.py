import numpy as np
from models import *
import random
from utils.Training_utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def cwloss(output, target, confidence=50, num_classes=10):
    # Compute the probability of the label class versus the maximum other
    # The same implementation as in repo CAT https://github.com/sunblaze-ucb/curriculum-adversarial-training-CAT
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = target_onehot.detach()
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss


def pgd(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init, class_num):
    model.eval()
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output, target, num_classes=class_num)
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.zero_grad()
    return x_adv.detach()


import torch.optim as optim


def trades_mixed(model, data, target, epsilon, num_steps, loss_fn, category, rand_init, Dataset, delta_ratio,
                 step_size=0.015, eps_con=0.2):
    model.eval()
    batch_size = len(data[0])

    # --- Core modification 1: Move clean sample prediction out of loop and use detach() to break gradient graph ---
    with torch.no_grad():
        clean_logits = model(data[0], data[1])
        # PyTorch KLDivLoss target must be probability distribution and must not have gradients
        clean_probs = F.softmax(clean_logits, dim=1).detach()

        # Initialize adversarial perturbation for continuous features
    adv_con = data[0] + delta_ratio * torch.randn(data[0].shape).cuda().detach()
    adv_con = adv_con.detach()

    # Initialize adversarial perturbation for categorical features
    delta_cat = 0.001 * torch.randn(data[1].shape).cuda().detach()
    delta_cat.requires_grad_(True)

    # Only manage optimizer for delta_cat
    optimizer_delta = optim.Adam([delta_cat], lr=epsilon / num_steps * 2)

    valid_mat = valid_mat_(Dataset, model)

    for _ in range(num_steps):
        # Rebuild clean leaf node each iteration to ensure gradient calculation
        adv_con = adv_con.detach().clone().requires_grad_(True)
        adv_cat = data[1] + delta_cat

        # --- Core modifications 2 & 3: Combine forward and backward, eliminate redundant calculations ---
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            output_adv = model(adv_con, adv_cat)

            # TRADES attack objective is to maximize KL divergence between adversarial and clean sample predictions
            loss_kl = nn.KLDivLoss(reduction='sum')(
                F.log_softmax(output_adv, dim=1),
                clean_probs
            )

        # Just 1 backward call gives gradients for both adv_con and delta_cat
        loss_kl.backward()

        # --- Update adv_con (continuous features) ---
        # Goal is to maximize loss_kl, so we move along gradient direction (+ step_size)
        grad_con = adv_con.grad.detach()
        adv_con = adv_con.detach() + step_size * torch.sign(grad_con)
        adv_con = torch.min(torch.max(adv_con, data[0] - eps_con), data[0] + eps_con)

        # --- Update delta_cat (categorical features) ---
        # Adam optimizer defaults to "minimize" loss. To make it "maximize" KL divergence,
        # we invert gradients before step (equivalent to (-1) * loss_cat in original code)
        delta_cat.grad = -delta_cat.grad

        # Renormalize gradient for delta_cat
        grad_norms_cat = delta_cat.grad.view(batch_size, -1).norm(p=1, dim=1)
        # --- Core modification 4: Add 1e-8 to prevent division by 0 causing NaN crash ---
        delta_cat.grad.div_(grad_norms_cat.view(-1, 1, 1) + 1e-8)

        # Avoid nan or inf if gradient is 0
        if (grad_norms_cat == 0).any():
            delta_cat.grad[grad_norms_cat == 0] = torch.randn_like(
                delta_cat.grad[grad_norms_cat == 0]
            )
        optimizer_delta.step()

        # Projection for delta_cat
        delta_cat.data = delta_cat.data * valid_mat
        delta_cat.data.add_(data[1])
        delta_cat.data.clamp_(0, 1).sub_(data[1])
        delta_cat.data.renorm_(p=1, dim=0, maxnorm=epsilon)

    # Final adversarial samples
    adv_cat = (data[1] + delta_cat).detach()
    adv_cat = torch.clamp(adv_cat, 0.0, 1.0).detach()

    model.zero_grad()
    return adv_con.detach(), adv_cat.detach()


def trades(model, data, target, epsilon, step_size, num_steps, loss_fn, category, rand_init, Dataset, delta_ratio):
    model.eval()
    batch_size = len(data)

    # --- Optimization point 1: Move clean sample prediction out of loop and disconnect gradient computation graph ---
    with torch.no_grad():
        clean_probs = F.softmax(model(data), dim=1).detach()

    # Generate adversarial example with l1 norm
    delta = delta_ratio * torch.randn(data.shape).cuda().detach()
    delta.requires_grad_()

    # Setup optimizers
    optimizer_delta = optim.Adam([delta], lr=epsilon / num_steps * 2)

    valid_mat = valid_mat_(Dataset, model)
    for _ in range(num_steps):
        x_adv = data + delta

        # Optimize
        optimizer_delta.zero_grad()
        with torch.enable_grad():
            output = model(x_adv)
            # --- Optimization point 2: Directly use precomputed clean_probs as target ---
            # Multiply by (-1) because Adam minimizes by default, but we want to maximize KL divergence
            loss = (-1) * nn.KLDivLoss(reduction='sum')(
                F.log_softmax(output, dim=1),
                clean_probs
            )
        loss.backward()

        # Renormalize gradient
        grad_norms = delta.grad.view(batch_size, -1).norm(p=1, dim=1)

        # --- Optimization point 3: Add 1e-8 to prevent division by 0 resulting in NaN ---
        delta.grad.div_(grad_norms.view(-1, 1, 1) + 1e-8)

        # Avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
        optimizer_delta.step()

        # Projection
        delta.data = delta.data * valid_mat
        delta.data.add_(data)
        delta.data.clamp_(0, 1).sub_(data)
        delta.data.renorm_(p=1, dim=0, maxnorm=epsilon)

    x_adv_final = (data + delta).detach()
    x_adv_final = torch.clamp(x_adv_final, 0.0, 1.0).detach()
    model.zero_grad()

    return x_adv_final


def eval_clean(model, test_loader, Dataset, class_num):
    model.eval()
    test_loss = 0
    correct = 0
    class_wise_correct = []
    class_wise_num = []
    # New: Collect all sample true labels and prediction probabilities (for AUC calculation)
    y_true = []
    y_pred = np.array([])
    y_prob = []
    for i in range(class_num):
        class_wise_correct.append(0)
        class_wise_num.append(0)
    with torch.no_grad():
        for batch_diagnosis_codes, target in test_loader:
            # data, target = data.cuda(), target.cuda()
            target = target.cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            if Dataset_type[Dataset] == 'mixed':  # [continuous numerical features, categorical features]
                output = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
            else:
                output = model(t_diagnosis_codes)

            test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            target = target.view_as(pred)
            eq_mat = pred.eq(target)

            for i in range(class_num):
                class_wise_num[i] += (target == i).int().sum().item()
                class_wise_correct[i] += eq_mat[target == i].sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()
            # --- New logic: Collect true labels and prediction probabilities ---
            # Convert labels and probabilities to CPU numpy arrays
            y_true.extend(target.cpu().numpy())  # True labels (shape: [batch_size])
            y_prob.extend(F.softmax(output, dim=1).detach().cpu().numpy())  # Prediction probabilities (shape: [batch_size, num_classes])
            prediction = torch.max(output, 1)[1].view((len(target),)).data.cpu().numpy()
            y_pred = np.concatenate((y_pred, prediction))

    # --- Calculate AUC ---
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc_score = compute_auc(y_true, y_prob, Dataset)

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)

    for i in range(class_num):
        class_wise_correct[i] /= class_wise_num[i]
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    return test_loss, test_accuracy, class_wise_correct, auc_score, precision_macro, recall_macro, f1_macro


def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, rand_init, Dataset, args, log_attack, class_num=10):  # Positional parameters (without default values) come first, keyword parameters (with default values) come after.
    model.eval()
    test_loss = 0
    correct = 0
    class_wise_correct = []
    class_wise_num = []
    for i in range(class_num):
        class_wise_correct.append(0)
        class_wise_num.append(0)

    # Initialize OMPGS attack related parameters
    attack_success = 0
    total_samples = 0
    # Initialize counters for the attack results
    robust_success = 0
    pred_right = 0
    # Initialize storage variables
    true_labels = []
    attack_probs = []
    y_pred = np.array([])
    from torch.utils.data import random_split
    total_test_samples = len(test_loader.dataset)
    if Dataset == 'UC_multi' or Dataset == 'UC_mixed':
        num_samples = int(total_test_samples)  # 0.015  0.01 *  0.01 *
    else:
        num_samples = int(total_test_samples * 0.01)  # 0.015  0.01 *  0.01 *
    print(f"Total test samples: {total_test_samples}, Number of samples to attack: {num_samples}")

    subset, _ = random_split(test_loader.dataset, [num_samples, total_test_samples - num_samples])
    subset_loader = torch.utils.data.DataLoader(subset, batch_size=test_loader.batch_size, shuffle=False)
    for batch_diagnosis_codes, target in subset_loader:
        # data, target = data.cuda(), target.cuda()
        target = target.cuda()
        if Dataset in ['cifar10', 'cifar100']:
            with torch.enable_grad():
                t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
                x_adv = pgd(model, t_diagnosis_codes, target, epsilon, step_size, perturb_steps, loss_fn, category, rand_init=rand_init, class_num=class_num)
        else:
            if Dataset in complex_categories.keys():
                attacker_myown = Attacker_mixed(model, log_attack, Dataset, epsilon)
            else:
                attacker_myown = Attacker(model, log_attack, Dataset, epsilon)
            # Table data specific processing path
            x_adv = []

            for i in range(len(target)):
                sample = batch_diagnosis_codes[i].cpu().numpy()
                # sample = batch_diagnosis_codes[i].detach().clone().requires_grad_(True)
                # Ensure if it's categorical data, convert to integer type
                if Dataset in complex_categories.keys() or Dataset_type[Dataset] == 'multi':
                    sample = sample.astype(int)  # Ensure categorical features are integer type
                label = target[i].item()

                if category == "OMPGS":
                    # Execute OMPGS attack
                    g_process, mf_process, _, _, _, robust_flag, \
                        _, _, greedy_set_best_temp_funccall, _, _, _, _ = attacker_myown.attack(sample, label)
                if category == "FSGS":
                    # Execute FSGS attack
                    g_process, mf_process, _, _, _, robust_flag, \
                        _, _, greedy_set_best_temp_funccall, _, _, _, _ = attacker_myown.attack_FSGS(sample, label)

                # Convert back to tensor and record results
                if robust_flag == 1:
                    # adv_sample = torch.from_numpy(greedy_set_best_temp_funccall).float()
                    # prob = attacker_myown.classify_prob(greedy_set_best_temp_funccall, label)
                    adv_sample = batch_diagnosis_codes[i]
                    prob = attacker_myown.classify_prob(sample, label)
                    robust_success += 1
                else:
                    # adv_sample = batch_diagnosis_codes[i]
                    # prob = attacker_myown.classify_prob(sample, label)
                    adv_sample = torch.from_numpy(greedy_set_best_temp_funccall).float()
                    prob = attacker_myown.classify_prob(greedy_set_best_temp_funccall, label)

                x_adv.append(adv_sample)
                total_samples += 1
                # Use squeeze() to remove extra dimensions
                prob = prob.squeeze()  # This converts (1, 3) to (3,)
                true_labels.append(label)
                attack_probs.append(prob)
            # Regroup as batch data
            x_adv = torch.stack(x_adv).cuda()

        # Unified model prediction logic
        t_diagnosis_codes = input_process(x_adv, Dataset)
        if Dataset_type[Dataset] == 'mixed':  # [continuous numerical features, categorical features]
            output = model(t_diagnosis_codes[0], t_diagnosis_codes[1])
        else:
            output = model(t_diagnosis_codes)
        # Calculate loss and accuracy
        test_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
        pred = output.max(1, keepdim=True)[1]
        # Update statistics
        target = target.view_as(pred)
        eq_mat = pred.eq(target)
        for i in range(class_num):
            class_wise_num[i] += (target == i).int().sum().item()
            class_wise_correct[i] += eq_mat[target == i].sum().item()
        # correct += pred.eq(target.view_as(pred)).sum().item()
        correct += eq_mat.sum().item()
        prediction = torch.max(output, 1)[1].view((len(target),)).data.cpu().numpy()
        y_pred = np.concatenate((y_pred, prediction))

    attack_probs = np.array(attack_probs)
    true_labels = np.array(true_labels)
    # test_loss /= len(test_loader.dataset)
    # test_accuracy = correct / len(test_loader.dataset)
    test_loss /= len(subset_loader.dataset)
    test_accuracy = correct / len(subset_loader.dataset)

    # Calculate OMPGS specific metrics
    if Dataset not in ['cifar10', 'cifar100']:
        if category == "OMPGS":
            ompgs_acc = robust_success / len(subset_loader.dataset)
            ompgs_auc = compute_auc(true_labels, attack_probs, Dataset)
            print(f'OMPGS Attack Success Rate: {ompgs_acc:.8f}, AUC: {ompgs_auc:.8f}', file=log_attack, flush=True)  # If program runs for a long time (e.g. training model or background service), want real-time log updates, can force flush buffer
            print(f'OMPGS Attack Success Rate: {ompgs_acc:.8f}, AUC: {ompgs_auc:.8f}')
        if category == "FSGS":
            fsgs_acc = robust_success / len(subset_loader.dataset)
            fsgs_auc = compute_auc(true_labels, attack_probs, Dataset)
            print(f'FSGS Attack Success Rate: {fsgs_acc:.8f}, AUC: {fsgs_auc:.8f}', file=log_attack, flush=True)
            print(f'FSGS Attack Success Rate: {fsgs_acc:.8f}, AUC: {fsgs_auc:.8f}')
        attack_acc = ompgs_acc if category == "OMPGS" else fsgs_acc
        auc = ompgs_auc if category == "OMPGS" else fsgs_auc

    precision_macro = precision_score(true_labels, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(true_labels, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(true_labels, y_pred, average='macro')
    # Calculate per-class accuracy
    class_wise_acc = [correct / total if total > 0 else 0
                      for correct, total in zip(class_wise_correct, class_wise_num)]
    print(f'Test Loss: {test_loss:.8f}, Test Accuracy: {test_accuracy:.8f}, Class-wise Accuracy: {class_wise_acc}, AUC: {auc:.8f}, Precision: {precision_macro:.8f}, Recall: {recall_macro:.8f}, F1: {f1_macro:.8f}', file=log_attack, flush=True)
    print(f'Test Loss: {test_loss:.8f}, Test Accuracy: {test_accuracy:.8f}, Class-wise Accuracy: {class_wise_acc}, AUC: {auc:.8f}, Precision: {precision_macro:.8f}, Recall: {recall_macro:.8f}, F1: {f1_macro:.8f}')

    return test_loss, test_accuracy, class_wise_acc, attack_acc, auc, precision_macro, recall_macro, f1_macro   # class_wise_correct