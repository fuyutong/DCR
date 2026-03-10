import time
from itertools import combinations

import numpy as np
import torch
from advertorch.context import ctx_noparamgrad_and_eval
from utils.utils import *
import torch.nn.functional as F


EPS = 1e-12


def compute_W_new(inputs, logits):
    """Vectorized gradient computation (5-20x faster than original method)"""
    C = logits.size(1)
    outputs = logits.sum(dim=0)  # [C]

    # Build gradient output matrix (avoid explicit identity matrix construction)
    grad_outputs = torch.zeros_like(outputs)  # [C]
    grad_outputs[range(C)] = 1.0  # Diagonal is 1

    # Batch gradient computation
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]  # [C, B, K] (when input is continuous features)

    # Handle unused gradients (categorical feature case)
    if grads is None:
        return torch.zeros(*inputs.shape, C, device=logits.device)

    return grads.movedim(0, 2)  # [B, K, C]


def BetaLoss_new(inputs, logit, Dataset, scale=1e6):
    """Optimized loss function"""
    # Handle 'multi' type data: pure categorical data (one-hot encoded)
    if Dataset_type[Dataset] == 'multi':
        W = compute_W(inputs, logit)
    elif Dataset_type[Dataset] == 'mixed':
        # Handle 'mixed' type data: continuous + categorical features
        # Separate continuous and categorical features
        continuous_features, categorical_features = inputs
        # You could apply different regularizations or loss computation for each
        W_con = compute_W(continuous_features, logit)  # For continuous features
        W_cat = compute_W(categorical_features, logit)  # For categorical features
        W = torch.cat((W_con, W_cat), dim=1)  # Combine them
    else:
        raise ValueError("Unsupported Dataset Type")

    def get_smooth_loss(y_pred):
        B, C = y_pred.shape
        eye = torch.eye(C, device=y_pred.device)

        # Optimized matrix construction
        A = (
                torch.diag_embed(y_pred) -
                torch.bmm(y_pred.unsqueeze(2), y_pred.unsqueeze(1)) +
                eye * EPS
        )

        # Stable eigendecomposition
        e, U = torch.linalg.eigh(A)
        Sigma = torch.diag_embed(torch.sqrt(torch.clamp(e, min=1e-20)))

        # Combined matrix operations
        L = torch.bmm(U, Sigma)  # [B, C, C]
        B_matrix = torch.bmm(W, L)  # [B, K, C]

        # Covariance matrix calculation optimization
        C_matrix = torch.bmm(B_matrix.transpose(1, 2), B_matrix) + eye * EPS

        # Calculate only max eigenvalue (using power iteration approximation)
        max_eigen = power_iteration(C_matrix, n_iters=2)  # 2 iterations are sufficient
        return max_eigen.mean() * scale

    return get_smooth_loss(logit)


def power_iteration(A, n_iters=2):
    """Fast max eigenvalue approximation (<0.1% error)"""
    B, C, _ = A.shape
    v = torch.randn(B, C, 1, device=A.device)

    for _ in range(n_iters):
        v = torch.bmm(A, v)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    return torch.bmm(torch.bmm(v.transpose(1, 2), A), v).squeeze()


def compute_W(inputs, logits):  # Calculate gradient for a batch of features mapping. Input features (whether continuous or categorical) effect on each class
    """
    inputs: Numerical feature tensor (needs gradients)
    logits: Model output unnormalized predictions
    """
    C = logits.size(1)
    W_list = []
    for c in range(C):
        # Calculate gradient of c-th class logit wrt numerical features
        grad_c = torch.autograd.grad(
            outputs=logits[:, c].sum(),
            inputs=inputs,
            create_graph=True,  # Keep computation graph for 2nd order derivatives
            retain_graph=True
        )[0]
        # Don't compute gradients for categorical feature embeddings
        grad_c = grad_c.mean(dim=-1)  # Average across feature dimension
        W_list.append(grad_c.unsqueeze(-1))

    W = torch.cat(W_list, dim=-1)  # (B, K, C)
    return W # W computed via compute_W function represents gradient of input features wrt model output (logits). Specifically, each column of W corresponds to gradient info for each class, useful for subsequent smooth loss calculation


def BetaLoss(inputs, logit, Dataset, scale=1e6):
    # Handle 'multi' type data: pure categorical data (one-hot encoded)
    if Dataset_type[Dataset] == 'multi':
        W = compute_W(inputs, logit)
    elif Dataset_type[Dataset] == 'mixed':
        # Handle 'mixed' type data: continuous + categorical features
        # Separate continuous and categorical features
        continuous_features, categorical_features = inputs
        # Apply different regularizations or loss computation for each
        W_con = compute_W(continuous_features, logit)  # For continuous features
        W_cat = compute_W(categorical_features, logit)  # For categorical features
        W = torch.cat((W_con, W_cat), dim=1)  # Combine them
    else:
        raise ValueError("Unsupported Dataset Type")

    def get_smooth_loss(y_pred):
        B, C = y_pred.shape  # B: batch size, C: number of classes
        A = torch.diag_embed(y_pred)  # (B, C, C)  # Create diagonal matrix A with shape (B, C, C). Diagonal elements are y_pred values representing model predictions
        p = y_pred.unsqueeze(2)  # (B, C, 1)  # Expand y_pred along last dimension to shape (B, C, 1)
        A = A - torch.matmul(p, p.transpose(1, 2)) # This is core of smooth geometry formula in paper # A - pp^T calculates outer product of p and transpose, result is (B, C, C) matrix  # Update A to remove correlations between predictions
        A = A + torch.eye(C, device=A.device).unsqueeze(0) * EPS # Create identity matrix and expand first dimension to shape (1, C, C) # EPS is small positive number for numerical stability preventing singular matrices # Add identity matrix times EPS to A to ensure A is positive definite
        e, U = torch.linalg.eigh(A)  # Eigen decomposition # Eigendecompose matrix A getting eigenvalues e and eigenvectors U  # Added in PyTorch 1.8.0, complexity O(n^3), slow
        e = torch.clamp(e, min=1e-20) # Use torch.clamp to limit eigenvalues e to minimum 1e-20 to avoid numerical issues
        Sigma = torch.diag_embed(torch.sqrt(e)) # Sigma is diagonal matrix with square roots of eigenvalues
        L = torch.matmul(U, Sigma) # Calculate smoothing matrix L through matrix multiplication, L = UΣ (B, C, C)
        B = torch.matmul(W, L)  # (B, K, C) x (B, C, C) --> (B, K, C)  # W shape (B, K, C), matrix mult gives (B, K, C) matrix B  # Reflects relationship between features and classes and plays key role in covariance calculation
        B_T = B.transpose(1, 2)  # B_T is transpose of B (B, C, K) dims 1 and 2 swapped
        C = torch.matmul(B_T, B)  # (B, C, C) Calculate covariance matrix C through matrix multiplication C = B^TB
        C = C + torch.eye(C.shape[1], device=C.device).unsqueeze(0) * EPS # Add small regularization term to covariance matrix C for numerical stability
        H_e, _ = torch.linalg.eigh(C) # Eigendecompose covariance matrix C getting eigenvalues H_e
        ave_beta = torch.mean(H_e[:, -1])  # Last eigenvalue  # Take average of last eigenvalue, typically related to max variance of matrix
        return ave_beta * scale  # scale is s in paper

    def smoothCE_loss(y_pred):
        # y_true: B, C, y_pred: B, C

        smoothness = get_smooth_loss(y_pred)
        return smoothness  # + (1 - alpha) * target_loss

    smooth_loss = smoothCE_loss(logit)  # Calculate loss
    return smooth_loss


from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np


def compute_auc(y_true, y_prob, Dataset):
    """Calculate AUC according to dataset type (consistent with training logic)"""
    if Dataset not in ['Splice', 'Thyroid_multi', 'Thyroid_multi_balanced','Thyroid_mixed','cardio_multi','cardio_mixed','diatri_mixed','diatri_multi']:  # Binary classification
        return roc_auc_score(y_true, y_prob[:, 1])  # Take positive class probability
    else:  # Multi-class
        # n_classes = len(np.unique(y_true))
        y_true_bin = label_binarize(y_true, classes=range(num_classes[Dataset]))  # Binarize labels
        return roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')


class Attacker(object):
    def __init__(self, model, log_f, Dataset, epsilon):
        # Classes of dataset
        self.n_labels = num_classes[Dataset]
        self.model = model
        # We only test data, so use this
        self.model.eval()
        # Log file
        self.log_f = log_f
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        # Number of categories in dataset
        self.n_diagonosis_codes = num_category[Dataset]
        self.Dataset = Dataset
        self.budgets = epsilon

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)
        # Below is what claude code added. Now I've found the root cause. Error occurred in utils/Training_utils.py input_handle function when processing table data samples. Sample
        # is float type (like numpy.float32), but code attempts to convert to LongTensor (integer type).

        # Problem analysis:
        # 1. sample from batch_diagnosis_codes[i].cpu().numpy(), might be float type values
        # 2. In input_handle function, code attempts to convert floats directly to LongTensor: torch.LongTensor([funccall])
        # 3. This stricter rejection of float-to-int conversion in newer PyTorch/NumPy versions

        # Solution:
        # Add appropriate type handling in input_handle function:

        # Check and convert data types
        if isinstance(funccall, (int, np.integer)):
            funccall = int(funccall)
        elif isinstance(funccall, (float, np.floating)):
            funccall = int(funccall)  # Or determine how to handle floats as needed
        elif isinstance(funccall, np.ndarray):
            funccall = funccall.astype(int)  # Ensure array elements are integer type

        # Put funccall and label in list
        funccall = torch.LongTensor([funccall])
        # Change list to one hot vectors
        t_diagnosis_codes = input_process(funccall, self.Dataset)
        # t_diagnosis_codes.requires_grad_(True)  # Explicitly enable gradients
        return t_diagnosis_codes

    def classify(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes)
        logit = logit.cpu()
        # Get prediction
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        # Get false labels
        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        g = logit[0][y]   # Row 0, because only one sample, y column is predicted label class
        # Find largest prediction in false labels
        h = max([logit[0][false_class] for false_class in list_label_set])
        return pred, g, h

    def classify_prob(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes)
        # New return logits or probabilities
        prob = logit.data.cpu().numpy()
        return prob  # Return prediction probabilities

    def eval_object(self, eval_funccall, greedy_set, orig_label, greedy_set_visit_idx, query_num, # Evaluate modification feature subset's impact on model output and select optimal modification scheme
                    greedy_set_best_temp_funccall):
        candidate_lists = []
        success_flag = 1
        funccall_lists = []
        # Get false labels
        label_set = set(range(self.n_labels))
        label_set.remove(orig_label)
        list_label_set = list(label_set)
        flip_set = set()
        flip_funccall = torch.tensor([])

        # candidate_lists contains all non-empty subsets of greedy_set
        for i in range(0, min(len(greedy_set) + 1, self.budgets)):  # Was budgets[self.Dataset], budgets[self.Dataset] limits subset size to avoid large computation, budgets has 5, meaning max 5 adversarial feature modification combinations
            subset1 = combinations(greedy_set, i)  # Generate all possible non-empty subsets from "greedy set"
            for subset in subset1:
                candidate_lists.append(list(subset))

        # Change funccall based on above candidates and get candidate funccalls.
        for can in candidate_lists:
            temp_funccall = copy.deepcopy(eval_funccall)
            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)
        query_num += len(funccall_lists)
        batch_size = 2 * batch_sizes[self.Dataset]  # 64
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))
        max_subsets_object = -np.inf
        max_subset_index = -1
        grad_feature_list = torch.tensor([])
        grad_cate_index_list = torch.tensor([], dtype=torch.long)
        # First, evaluate all candidates get gradients, then find largest gradient candidate and category for each feature
        # Evaluate all candidates, get gradients, then find largest gradient candidate and category
        for index in range(n_batches):  # n_batches
            self.model.eval()
            batch_diagnosis_codes = torch.LongTensor(funccall_lists[batch_size * index: batch_size * (index + 1)])
            t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
            logit = self.model(t_diagnosis_codes)
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]  # Original max class for each sample
            subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
            subsets_objects = subsets_h - subsets_g
            max_subset_object_temp = max(subsets_objects)
            if max_subset_object_temp > max_subsets_object:
                max_subsets_object = max_subset_object_temp
                max_subset_index = batch_size * index + np.argmax(subsets_objects)

            self.model.train()
            self.model.apply(fix_bn)

            grad_all = torch.tensor([])
            flag = 0  # If one sample's gradient calculation fails, flag is set to 1
            for i in range(len(list_label_set)): # Why calculate for each false class? Because we want to find max gradient for each false class  # Iterate each class label in list_label_set Since only 2 classes, runs once, check Splice case again
                flag = 0  # Purpose of this line is create batch of same labels, all same as current false class list_label_set[i]
                self.model.zero_grad()  # Before each backprop, usually clear gradients to avoid accumulating prev gradients
                batch_labels = torch.tensor([list_label_set[i]] * len(batch_diagnosis_codes)).cuda() #[1]*1=[1] # Access list_label_set[i] element, wrap in brackets to create new list with just that element
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                if t_diagnosis_codes.size(0) == 1:  # If only one sample  Only first attack for each sample does this
                    flag = 1
                    if Dataset_type[self.Dataset] == 'multi':
                        t_diagnosis_codes = t_diagnosis_codes.repeat(2, 1, 1) # Copy once  # First param 2 means repeat 2x in first dim (sample count), 2nd and 3rd params 1 means keep other dims unchanged
                    else:
                        t_diagnosis_codes = t_diagnosis_codes.repeat(2, 1)  # t_diagnosis_codes only copied in first dim, second dim unchanged. Usually for single-label classification.
                    batch_labels = batch_labels.repeat(2) # Copy once [1]->[1, 1]
                # t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)
                t_diagnosis_codes.requires_grad_()
                logit = self.model(t_diagnosis_codes)
                loss = self.criterion(logit, batch_labels)
                loss.backward()
                # we use the gradient of the false label. since there are only 3 lables, we just use grad_0 and _1 # Translation: we use false label's gradient. Since only 3 labels, only use grad_0 and _1
                grad = t_diagnosis_codes.grad.cpu().data # Get t_diagnosis_codes tensor's gradient, move to CPU, access raw data
                # for Splice, there is a invalid category, and we need to remove it.
                grad = torch.abs(grad)
                # print(grad_0[:, 0].norm(dim=0))
                grad_all = torch.cat((grad_all, grad.unsqueeze(0)), dim=0) # grad_all tensor to store each sample's gradient info. Specifically, each element of grad_all is gradient tensor representing one sample's gradient info

            self.model.zero_grad() # Clear gradients
            grad = torch.max(grad_all, dim=0)[0] # Return tuple with two elements: first is max value tensor along specified dim (here dim 0). Second is indices of max values. Only take first element, max value tensor
            if flag == 1: # If copied before and only one sample, marked as 1
                grad = grad[0].unsqueeze(0)  # (10,6)->(1,10,6)
            subsets_g = subsets_g.reshape(-1, 1)
            subsets_g = torch.tensor(subsets_g)
            if Dataset_type[self.Dataset] == 'multi':
                grad_feature_temp = torch.max(grad, dim=2)[0] # Max value of one sample's one feature across 6 possible categorical values
                grad_feature_temp = grad_feature_temp / subsets_g # Divide by original prediction value
                grad_cate_index = torch.argmax(grad, dim=2) # Get index of max gradient value
                grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=0)  #
            else:
                grad_feature_temp = grad / subsets_g
            grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=0)  # Concatenate along dim 0

        # If one of candidates attacks successfully, then we exit.
        if max_subsets_object >= 0 or len(greedy_set) == num_feature[self.Dataset]:  # If max subset objective >=0, or greedy set size equals feature count
            if max_subsets_object >= 0:  # Attack succeeded
                # print(max_subsets_object)
                success_flag = 0
                flip_funccall = copy.deepcopy(funccall_lists[max_subset_index])
                greedy_set_best_temp_funccall = copy.deepcopy(funccall_lists[max_subset_index])
                flip_set = self.changed_set(eval_funccall, flip_funccall)
            else:  # Attack unsuccessful, but greedy set size equals feature count, means all possible features attacked, like 10 features all attacked but unsuccessful, then return -2
                # success flag = -2 means we have attacked all features.
                success_flag = -2
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                greedy_set_visit_idx, flip_set, flip_funccall, query_num

        self.model.eval()
        grad_feature, grad_set_index_list = torch.max(grad_feature_list, dim=0) #Take max along dim 0 (row direction), first attack since only one row, therefore max of each col is value itself. Index is all 0, because only one row per col
        top_100_features = torch.argsort(grad_feature, descending=True)[:100]  # Sort top hundred indices from large to small
        funccalls = []
        features = []
        # for each feature, we choose the optimal candidate and optimal category and then we run the exactly and pick the largest.
        # According to above code calc each feature, start from largest gradient feature modifying each feature
        for index in top_100_features:   # Loop from large to small
            if index.item() in greedy_set_visit_idx:  # If this feature visited before, skip
                continue
            temp_funccall = copy.deepcopy(funccall_lists[grad_set_index_list[index]])  # See from which sample # Initialize temp_funccall each time, take from funccall_lists corresponding to grad_set_index_list[7]=0. Initialize temp_funccall to help with positioning, if >64 features, goes to next batch, will have second row?
            if Dataset_type[self.Dataset] == 'multi':
                temp_funccall[index] = int(grad_cate_index_list[grad_set_index_list[index], index].item()) # Row-col positioning  # grad_cate_index_list[0,7].item = 2 First 0 doesn't matter, because grad_cate has 2 levels, taking first 0 just opens one level
                if self.Dataset in complex_categories.keys():  # If data is mixed type
                    if temp_funccall[index] >= complex_categories[self.Dataset][index]:
                        print('!!!', index, temp_funccall[index], '!!!')
                        continue
            elif Dataset_type[self.Dataset] == 'binary':
                temp_funccall[index] = 1 - temp_funccall[index]
            else:
                pass
            features.append(index)
            funccalls.append(temp_funccall)

        funccalls = torch.LongTensor(funccalls)
        query_num += len(features)   # Each loop over 100 attackable features, add one query, but stroke dataset only has 10 features, so added 10 queries
        t_diagnosis_codes = input_process(funccalls, self.Dataset)
        # t_diagnosis_codes = torch.tensor(t_diagnosis_codes).cuda()
        logit = self.model(t_diagnosis_codes)
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
        objects = h - g

        max_object = np.max(objects)
        max_index = np.argmax(objects)

        max_feature = features[max_index].item()
        if Dataset_type[self.Dataset] == 'multi':
            max_category = grad_cate_index_list[grad_set_index_list[max_feature], max_feature].item() # grad_cate_index_list[0,1].item() = 2 Want to change feature 7 to category 2
        elif Dataset_type[self.Dataset] == 'binary':
            max_category = int(1 - eval_funccall[max_feature])
        else:
            max_category = None

        # if the max object changs, we update it and the best funccall
        if max_object < max_subsets_object:  # Prev best was -0.75, this time -0.77, not bigger than prev
            max_object = max_subsets_object
            greedy_set_best_temp_funccall = funccall_lists[max_subset_index]  # Means keep prev result
        else:
            max_set = grad_set_index_list[max_feature]  # max_feature = 1 max_set: tensor(0)
            greedy_set_best_temp_funccall = copy.deepcopy(funccall_lists[max_set])
            greedy_set_best_temp_funccall[max_feature] = max_category # Put max gradient corresponding feature and category in greedy_set_best_temp_funccall, greedy is historical optimal feature set

        if max_object >= 0:
            success_flag = 0
            flip_funccall = greedy_set_best_temp_funccall   # Is optimal, success means assign to flip, flip is final successful adversarial sample
            flip_set = self.changed_set(eval_funccall, flip_funccall)

        # update the greedy set
        greedy_set_visit_idx.add(max_feature)   # Although not biggest, still put 2nd attack's changed feature combo in greedy_set return
        greedy_set.add((max_feature, max_category))

        return max_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
            flip_set, flip_funccall, query_num

    # calculate which feature is changed
    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall != new_funccall)[0])
        return diff_set

    def attack(self, funccall, y):
        # print()
        st = time.time()
        success_flag = 1  # 0: Attack succeeded. 1: Attack not yet succeeded. -2: Tried all features but attack still unsuccessful.

        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0  # Attack count
        robust_flag = 0
        query_num = 0  # Query count accumulated per attack

        current_object = orig_h - orig_g
        flip_funccall = funccall
        time_limit = OMPGS_time_limits[self.Dataset]

        # when the classification is wrong for the original example, skip the attack # If original sample classification error, skip attack
        if current_object > 0:
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
        # once the success_flag ==0, we attack successfully and exit
        while success_flag == 1:
            iteration += 1
            # Now problem is clear! funccall_lists contains data copied from temp_funccall, and temp_funccall copied from initial funccall (ie sample)
            #   problem is, eval_funccall param (ie original sample) is numpy.float32 type, causing entire funccall_lists list contain floats not
            #   integers.
            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
                flip_set, flip_funccall, query_num = self.eval_object(funccall, greedy_set, y,
                                                                      greedy_set_visit_idx, query_num,
                                                                      greedy_set_best_temp_funccall)

            # print(iteration)
            # print(worst_object)
            # print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))  # Still compare diff with best case
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(worst_object)
            greedy_set_process.append(copy.deepcopy(greedy_set))

            # time limit exceed or we have attacked all features, but it is still not successful.
            # if (time.time() - st) > time_limit or success_flag == -2:
            if iteration == self.budgets or success_flag == -2:   # Find five adversarial samples for one original sample
                success_flag = -1  # Means attack ended
                robust_flag = 1  # Means not yet attacked down,

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration

    def funccall_query(self, eval_funccall, greedy_set):
        candidate_lists = []
        funccall_lists = []

        for i in range(min(len(greedy_set) + 1, self.budgets)):
            subset1 = combinations(greedy_set, i)
            for subset in subset1:
                candidate_lists.append(list(subset))

        # change the funccall based on the candidates above and get the candidate funccalls.
        for can in candidate_lists:
            temp_funccall = copy.deepcopy(eval_funccall)
            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)

        return funccall_lists

    def attack_FSGS(self, funccall, y):
        st = time.time()
        time_limit = 1
        success_flag = 1
        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0
        query_num = 0

        current_object = orig_h - orig_g
        flip_funccall = funccall

        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        # when the classification is wrong for the original example, skip the attack
        if current_object > 0:
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
        # once the success_flag ==0, we attack successfully and exit
        while success_flag == 1:
            iteration += 1
            success_flag = 1
            pos_dict = {}
            funccall_lists_all = []

            # we loop over each feature and each category to find the worst object and its position
            for visit_idx in range(len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                for code_idx in range(num_avail_category[self.Dataset]):
                    if code_idx == funccall[visit_idx]:
                        continue
                    if self.Dataset in complex_categories.keys():
                        if code_idx >= complex_categories[self.Dataset][visit_idx]:
                            break
                    pos = (visit_idx, code_idx)
                    eval_funccall = copy.deepcopy(funccall)
                    eval_funccall[visit_idx] = code_idx
                    funccall_list_temp = self.funccall_query(eval_funccall, greedy_set)

                    funccall_lists_all = funccall_lists_all + funccall_list_temp
                    pos_dict[len(funccall_lists_all)] = pos

            query_num += len(funccall_lists_all)
            batch_size = 2 * batch_sizes[self.Dataset] # 512
            n_batches = int(np.ceil(float(len(funccall_lists_all)) / float(batch_size)))
            max_object = -np.inf
            max_index = 0
            for index in range(n_batches):  # n_batches

                batch_diagnosis_codes = torch.LongTensor(
                    funccall_lists_all[batch_size * index: batch_size * (index + 1)])
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                logit = self.model(t_diagnosis_codes)
                logit = logit.data.cpu().numpy()
                subsets_g = logit[:, y]
                subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
                subsets_object = subsets_h - subsets_g
                # get the maximum object, and update worst object
                temp_max_object = np.max(subsets_object)
                temp_max_index = np.argmax(subsets_object) + batch_size * index

                if temp_max_object > max_object:
                    max_object = temp_max_object
                    max_index = temp_max_index
            poses = np.array(list(pos_dict.keys()))
            max_pos_index = np.where(poses > max_index)[0][0]
            max_pos = pos_dict[poses[max_pos_index]]
            greedy_set_best_temp_funccall = funccall_lists_all[max_index]

            # print(iteration)
            # print('query', query_num)
            # print(max_object)

            greedy_set.add(max_pos)
            greedy_set_visit_idx.add(max_pos[0])
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))
            if max_object > current_object:
                current_object = max_object
            if max_object > 0:
                success_flag = 0
            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))

            # print(greedy_set)
            if success_flag == 1:
                # if (time.time() - st) > time_limit or len(greedy_set) == num_feature[self.Dataset]:
                if iteration == self.budgets or len(greedy_set) == num_feature[self.Dataset]:
                    success_flag = -1
                    robust_flag = 1
                    # print('Time out')

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)
            # print('Attack successfully')
        #
        # print("Modified_set:", flip_set)
        # print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration


class Attacker_mixed(object):
    def __init__(self, model, log_f, Dataset, epsilon):
        # the classes of the dataset
        self.n_labels = num_classes[Dataset]
        self.model = model
        # We only test data, so use this
        self.model.eval()
        # the log file
        self.log_f = log_f
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # the number of the categories of the dataset
        self.n_diagonosis_codes = num_category[Dataset]
        self.Dataset = Dataset
        self.n_con_fea = num_con_feature[Dataset]
        self.budgets = epsilon
        self.adversary = LinfPGDAttack_mixed(self.model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2, # # Key adversarial generation module
                                                 nb_iter=30, eps_iter=0.02, rand_init=False, clip_min=-np.inf,
                                                 clip_max=np.inf, targeted=False)
        # LinfPGDAttack_mixed is core part of attacker, responsible for calculating gradients based on model predictions and performing adversarial perturbations. Control attack strength and iteration count via eps, nb_iter params
        # eps=0.2: Max norm of adversarial perturbation Max L∞ norm constraint (allowable pixel value change range)
        # nb_iter=30: PGD iteration count
        # eps_iter=0.02: Perturbation size per iteration

    def input_handle(self, funccall):  # input:funccall, output:(seq_len,n_sample,m)  # Transform input format
        # put the funccall and label into a list
        funccall = torch.FloatTensor([funccall])
        # change the list to one hot vectors
        t_diagnosis_codes = input_process(funccall, self.Dataset)
        return t_diagnosis_codes

    def classify(self, funccall, y): # Calculate original classification result of input sample (input is single sample), get model's prediction and probability difference with original label (orig_g and orig_h)
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall) # Convert input sample to model input format
        logit = self.model(weight_of_embed_codes[0], weight_of_embed_codes[1]) # Get model's prediction result for input sample, input x(funccall) has numerical and categorical parts
        logit = logit.cpu()
        # get the prediction
        pred = torch.max(logit, 1)[1].view((1,)).data.numpy()
        logit = logit.data.cpu().numpy()
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(y)  # Remove this single sample's correct category
        list_label_set = list(label_set)
        g = logit[0][y]  # Get correct category's prediction value, logit[0] is model's prediction for input sample, y is correct category
        # find the largest prediction in the false labels
        h = max([logit[0][false_class] for false_class in list_label_set]) # Get max prediction value of false categories
        return pred, g, h # Return prediction result, correct category's prediction value, max prediction value of false categories

    def classify_prob(self, funccall, y):
        self.model.eval()
        weight_of_embed_codes = self.input_handle(funccall)
        logit = self.model(weight_of_embed_codes[0], weight_of_embed_codes[1])
        # New return logits or probabilities
        prob = logit.data.cpu().numpy()
        return prob  # Return prediction probabilities

    def eval_object(self, eval_funccall, greedy_set, orig_label, greedy_set_visit_idx, query_num, # Evaluate and modify input
                    greedy_set_best_temp_funccall, adversary):
        # 1. Initialize
        candidate_lists = []  # Store all possible candidate feature subsets
        success_flag = 1  # Initialize to 1, means attack not yet succeeded
        funccall_lists = []  # Used to store modified function call sequences (adversarial samples) based on candidates
        # get the false labels
        label_set = set(range(self.n_labels))
        label_set.remove(orig_label)  # All possible label set excluding original label
        list_label_set = list(label_set)
        flip_set = set()   # Set to store indices of modified features
        flip_funccall = torch.tensor([])  # Final adversarial sample

        # 2. Generate candidate feature subsets
        # candidate_lists contains all the non-empty subsets of greedy_set  # Generate candidate subsets
        for i in range(0, min(len(greedy_set) + 1, self.budgets)):  # budgets[self.Dataset] limits subset size to avoid large computation
            subset1 = combinations(greedy_set, i)  # Generate all possible non-empty subsets from "greedy set"
            for subset in subset1:
                candidate_lists.append(list(subset))  # Convert each subset to list, add to candidate_lists

        # 3. Generate candidate adversarial samples
        # change the funccall based on the candidates above and get the candidate funccalls. # Generate candidate adversarial samples
        for can in candidate_lists:  # Iterate each candidate subset in candidate_lists
            temp_funccall = copy.deepcopy(eval_funccall)  # Copy input sample, init  For each subset, create deep copy of eval_funccall
            for position in can:
                visit_idx = position[0]  # Modify temp_funccall based on subset containing feature positions and values
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx  # Modify input sample, also new adversarial sample

            funccall_lists.append(temp_funccall)

        # 4. Evaluate candidate adversarial samples
        funccall_lists = np.array(funccall_lists)   # Adversarial sample set
        query_num += len(funccall_lists)  # Count of queries needed by set
        batch_size = 2 * batch_sizes[self.Dataset] # 64
        n_batches = int(np.ceil(float(len(funccall_lists)) / float(batch_size)))  # May have multiple adversarial samples, arrange by batch
        max_subsets_object = -np.inf  # Record currently found max objective function value
        max_subset_index = -1  # Record index of adversarial sample corresponding to max objective function value
        grad_feature_list = torch.tensor([])  # Gradient feature list, each new adversarial sample gradient category index
        grad_cate_index_list = torch.tensor([], dtype=torch.long)  # index list
        # first, we eval all the candidates and get the gradients, and then we find the largest gradient candidate and category for each feature
        # First, evaluate all candidates get gradients, then find max gradient candidate and category for each feature
        for index in range(n_batches):  # n_batches
            self.model.eval()
            batch_diagnosis_codes = torch.FloatTensor(funccall_lists[batch_size * index: batch_size * (index + 1)])
            batch_labels = torch.tensor([orig_label] * len(batch_diagnosis_codes)).cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
            # Force model to evaluation mode (Eval Mode)
            # Working principle:
            # Call model.eval(), disable Dropout, BatchNorm layer training mode behavior.
            # Why needed:
            # Result consistency: Evaluation mode uses fixed statistics (like BatchNorm's running_mean), ensure stable adversarial sample generation process
            # Avoid randomness: Disable random operations like Dropout, make attack result reproducible

            # Assume model is pretrained model, # Use context manager to wrap attack process
            with ctx_noparamgrad_and_eval(self.model):  # Ensure model parameters not modified
                # In this context, model's gradient calculation disabled, switched to eval mode
                t_diagnosis_codes[0] = adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1], batch_labels).detach()  # Perturb continuous features
                funccall_lists[batch_size * index: batch_size * (index + 1), :self.n_con_fea] = t_diagnosis_codes[0].cpu().numpy()
            logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])  # Calculate model's output logit for current batch adversarial samples
            logit = logit.data.cpu().numpy()
            subsets_g = logit[:, orig_label]
            subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
            subsets_objects = subsets_h - subsets_g
            max_subset_object_temp = max(subsets_objects)
            if max_subset_object_temp > max_subsets_object:
                max_subsets_object = max_subset_object_temp
                max_subset_index = batch_size * index + np.argmax(subsets_objects)  # np.argmax(subsets_objects) returns index of max gradient value

            self.model.train()
            self.model.apply(fix_bn)   # Set all BatchNorm layers in model to eval mode
            grad_all = torch.tensor([])
            flag = 0
            # Calculate gradients
            for i in range(len(list_label_set)):  # Iterate false categories Calculate model's gradient wrt current batch adversarial samples and false categories
                flag = 0
                self.model.zero_grad()
                batch_labels = torch.tensor([list_label_set[i]] * len(batch_diagnosis_codes)).cuda()
                # Added below, prevent error, check effect
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                if t_diagnosis_codes[1].size(0) == 1:
                    flag = 1
                    # if Dataset_type[Dataset] == 'multi':
                    t_diagnosis_codes[1] = t_diagnosis_codes[1].repeat(2, 1, 1)
                    t_diagnosis_codes[0] = t_diagnosis_codes[0].repeat(2, 1)
                    # else:
                    #     t_diagnosis_codes[1] = t_diagnosis_codes.repeat(2, 1)
                    batch_labels = batch_labels.repeat(2)
                # t_diagnosis_codes[1] = torch.autograd.Variable(t_diagnosis_codes[1].data, requires_grad=True)
                t_diagnosis_codes[1].requires_grad_()
                logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
                loss = self.criterion(logit, batch_labels)
                loss.backward()
                # we use the gradient of the false label. since there are only 3 lables, we just use grad_0 and _1
                # Use false category's gradient, since only 3 labels, only use grad_0 and _1
                grad = t_diagnosis_codes[1].grad.cpu().data
                # for Splice, there is a invalid category, and we need to remove it.
                grad = torch.abs(grad)
                # print(grad_0[:, 0].norm(dim=0))
                grad_all = torch.cat((grad_all, grad.unsqueeze(0)), dim=0)  # Add gradient to grad_all

            self.model.zero_grad()
            grad = torch.max(grad_all, dim=0)[0]
            if flag == 1: #
                grad = grad[0].unsqueeze(0)
            subsets_g = subsets_g.reshape(-1, 1)
            subsets_g = torch.tensor(subsets_g)
            # if Dataset_type[Dataset] == 'multi':
            grad_feature_temp = torch.max(grad, dim=2)[0]
            grad_feature_temp = grad_feature_temp / subsets_g
            grad_cate_index = torch.argmax(grad, dim=2)
            grad_cate_index_list = torch.cat((grad_cate_index_list, grad_cate_index), dim=0)
            # else:
            #     grad_feature_temp = grad / subsets_g
            grad_feature_list = torch.cat((grad_feature_list, grad_feature_temp), dim=0)  # Select max gradient, store in grad_feature_list and grad_cate_index_list

        # 5. Check if attack successful, if one of candidates successfully attacks, then exit
        # if the one of the candidates attacks successfully, then we exit.
        if max_subsets_object >= 0 or len(greedy_set) == num_feature[self.Dataset]:
            if max_subsets_object >= 0:
                # print(max_subsets_object)
                success_flag = 0  # Attack succeeded
                flip_funccall = copy.deepcopy(funccall_lists[max_subset_index])  # Best adversarial sample
                greedy_set_best_temp_funccall = copy.deepcopy(funccall_lists[max_subset_index])  # Best adversarial sample
                flip_set = self.changed_set(eval_funccall, flip_funccall)  # Calculate indices of modified features flip_set
            else:
                # success flag = -2 means we have attacked all the features.
                success_flag = -2  # Have tried all features, but attack still unsuccessful
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                greedy_set_visit_idx, flip_set, flip_funccall, query_num

        # 6. Select next feature to modify
        self.model.eval()
        grad_feature, grad_set_index_list = torch.max(grad_feature_list, dim=0)
        top_100_features = torch.argsort(grad_feature, descending=True)[:100]  # Select top 100 features with max gradients
        funccalls = []
        features = []
        # for each feature, we choose the optimal candidate and optimal category and then we run the exactly and pick the largest.
        # For each feature, we select optimal candidate and optimal category, then run exactly and pick largest.
        for index in top_100_features:
            if (index.item() + self.n_con_fea) in greedy_set_visit_idx:
                continue
            temp_funccall = copy.deepcopy(funccall_lists[grad_set_index_list[index]])
            # if Dataset_type[Dataset] == 'multi':
            if self.Dataset in complex_categories.keys():
                if grad_cate_index_list[grad_set_index_list[index], index] >= complex_categories[self.Dataset][index]:
                    continue
            temp_funccall[index + self.n_con_fea] = grad_cate_index_list[grad_set_index_list[index], index]
            # elif Dataset_type[Dataset] == 'binary':
            #     temp_funccall[index] = 1 - temp_funccall[index]
            features.append(index + self.n_con_fea)
            funccalls.append(temp_funccall)

        if not funccalls:  # If funccalls empty, means all possible modifications tried, set success_flag = -2 and return
            success_flag = -2
            return max_subsets_object, greedy_set_best_temp_funccall, success_flag, greedy_set, \
                greedy_set_visit_idx, flip_set, flip_funccall, query_num

        funccalls = torch.LongTensor(funccalls)  # Convert funccalls to PyTorch tensor
        query_num += len(features)
        t_diagnosis_codes = input_process(funccalls, self.Dataset)
        batch_labels = torch.tensor([orig_label] * len(funccalls)).cuda()
        with ctx_noparamgrad_and_eval(self.model):
            t_diagnosis_codes[0] = adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],   # Use adversary.perturb to perturb continuous features
                                                     batch_labels).detach()
            funccalls = funccalls.numpy()
            funccalls[:, :self.n_con_fea] = t_diagnosis_codes[0].cpu().numpy()
        logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
        logit = logit.data.cpu().numpy()

        g = logit[:, orig_label]
        h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
        objects = h - g   # Objective function value

        max_object = np.max(objects)
        max_index = np.argmax(objects)

        max_feature = features[max_index].item()
        # if Dataset_type[Dataset] == 'multi':
        max_category = grad_cate_index_list[   # Objective function value's max feature max_feature and category max_category
            grad_set_index_list[max_feature - self.n_con_fea], max_feature - self.n_con_fea].item()
        # elif Dataset_type[Dataset] == 'binary':
        #     max_category = int(1 - eval_funccall[max_feature])
        # else:
        #     max_category = None
        # if the max object changs, we update it and the best funccall
        if max_object < max_subsets_object:  # If max_object < max_subsets_object, update max_object and greedy_set_best_temp_funccall
            max_object = max_subsets_object
            greedy_set_best_temp_funccall = funccall_lists[max_subset_index]
        else:  # Otherwise, update greedy_set_best_temp_funccall
            greedy_set_best_temp_funccall = funccalls[max_index]

        if max_object >= 0:  # If max_object >= 0, means attack successful, set success_flag = 0, update flip_funccall and flip_set
            success_flag = 0
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(eval_funccall, flip_funccall)

        #8. Update greedy set
        greedy_set_visit_idx.add(max_feature)
        greedy_set.add((max_feature, max_category))

        return max_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
            flip_set, flip_funccall, query_num

    # calculate which feature is changed
    def changed_set(self, eval_funccall, new_funccall):
        diff_set = set(np.where(eval_funccall[self.n_con_fea:] != new_funccall[self.n_con_fea:])[0])
        return diff_set

    # OMPGS attack
    def attack(self, funccall, y):  # funccall is input sample, y is original label
        # print()
        success_flag = 1

        orig_pred, orig_g, orig_h = self.classify(funccall, y) # Get original classification result of input sample, original classification result's correct category prediction value and max prediction value of wrong categories

        # Greedy Set construction
        # Attack method finds combination of features and categories most likely to cause prediction failure through greedy search
        greedy_set = set() # Used to store already modified features and categories
        greedy_set_visit_idx = set() # Record already visited feature positions. By continuously updating these sets, attacker keeps optimizing input data
        greedy_set_best_temp_funccall = funccall # Used to store final attack result, initialize first to funccall
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0 # Attack success flag
        query_num = 0 # Model query count

        current_object = orig_h - orig_g # Difference between confidence of correct classification and max confidence of wrong classification
        flip_funccall = funccall
        time_limit = OMPGS_time_limits[self.Dataset]

        # when the classification is wrong for the original example, skip the attack
        if current_object > 0: # Means find adversarial sample that makes model classify wrong, attack process won't continue
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration

        # print(current_object)
        # once the success_flag ==0, we attack successfully and exit
        st = time.time()
        # Main loop, iterate until attack succeeds
        # Below eval_object function used to evaluate current feature and category combination, return combination of features and categories most likely to cause prediction failure, evaluate and modify input
        while success_flag == 1:
            iteration += 1  # Iterated several times to find adversarial sample
            # Evaluate candidate perturbation: evaluate current feature and category combination, return combination of features and categories most likely to cause prediction failure
            worst_object, greedy_set_best_temp_funccall, success_flag, greedy_set, greedy_set_visit_idx, \
                flip_set, flip_funccall, query_num = self.eval_object(funccall, greedy_set, y,
                                                                      greedy_set_visit_idx, query_num,
                                                                      greedy_set_best_temp_funccall, self.adversary)

            # print(iteration)
            # print(worst_object)
            # print(greedy_set)

            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall)) # Record modified features
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y) # Get classification result of modified sample
            g_process.append(float(g))  # Record correct category's prediction value
            mf_process.append(worst_object)     # Record max prediction value of wrong categories
            greedy_set_process.append(copy.deepcopy(greedy_set)) # Record modified features and categories

            # time limit exceed or we have attacked all the features, but it is still not successful.
            # if (time.time() - st) > time_limit or success_flag == -2:
            if iteration == self.budgets or success_flag == -2:  #Reached max allowed attack count
                success_flag = -1  # Attack failed, exit loop
                robust_flag = 1  # Still robust, not attacked down

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))  # After attack success, how many features modified

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration

    def funccall_query(self, eval_funccall, greedy_set):
        candidate_lists = []
        funccall_lists = []

        for i in range(min(len(greedy_set) + 1, self.budgets)):
            subset1 = combinations(greedy_set, i)
            for subset in subset1:
                candidate_lists.append(list(subset))

        # change the funccall based on the candidates above and get the candidate funccalls.
        for can in candidate_lists:
            temp_funccall = copy.deepcopy(eval_funccall)
            for position in can:
                visit_idx = position[0]
                code_idx = position[1]
                temp_funccall[visit_idx] = code_idx

            funccall_lists.append(temp_funccall)

        return funccall_lists

    def attack_FSGS(self, funccall, y):
        time_limit = 1
        success_flag = 1
        orig_pred, orig_g, orig_h = self.classify(funccall, y)

        greedy_set = set()
        greedy_set_visit_idx = set()
        greedy_set_best_temp_funccall = funccall
        flip_set = set()

        g_process = []
        mf_process = []
        greedy_set_process = []
        changed_set_process = []

        g_process.append(float(orig_g))
        mf_process.append(float(orig_h - orig_g))

        n_changed = 0
        iteration = 0
        robust_flag = 0
        query_num = 0

        current_object = orig_h - orig_g
        flip_funccall = funccall

        label_set = set(range(self.n_labels))
        label_set.remove(y)
        list_label_set = list(label_set)
        # when the classification is wrong for the original example, skip the attack
        if current_object > 0:
            robust_flag = -1
            # print("Original classification error")

            return g_process, mf_process, greedy_set_process, changed_set_process, \
                query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
                greedy_set_best_temp_funccall, \
                n_changed, flip_funccall, flip_set, iteration
        # print(current_object)
        # once the success_flag ==0, we attack successfully and exit
        st = time.time()
        while success_flag == 1:
            iteration += 1
            success_flag = 1
            pos_dict = {}
            funccall_lists_all = []

            # we loop over each feature and each category to find the worst object and its position
            for visit_idx in range(self.n_con_fea, len(funccall)):
                if visit_idx in greedy_set_visit_idx:
                    continue
                for code_idx in range(num_avail_category[self.Dataset]):
                    if code_idx == funccall[visit_idx]:
                        continue
                    if self.Dataset in complex_categories.keys():
                        if code_idx >= complex_categories[self.Dataset][visit_idx - self.n_con_fea]:
                            break
                    pos = (visit_idx, code_idx)
                    eval_funccall = copy.deepcopy(funccall)
                    eval_funccall[visit_idx] = code_idx
                    funccall_list_temp = self.funccall_query(eval_funccall, greedy_set)

                    funccall_lists_all = funccall_lists_all + funccall_list_temp
                    pos_dict[len(funccall_lists_all)] = pos

            query_num += len(funccall_lists_all)
            batch_size = 2 * batch_sizes[self.Dataset] # 512
            n_batches = int(np.ceil(float(len(funccall_lists_all)) / float(batch_size)))
            max_object = -np.inf
            max_index = 0
            for index in range(n_batches):  # n_batches

                batch_diagnosis_codes = torch.FloatTensor(
                    funccall_lists_all[batch_size * index: batch_size * (index + 1)])
                t_diagnosis_codes = input_process(batch_diagnosis_codes, self.Dataset)
                with ctx_noparamgrad_and_eval(self.model):
                    t_diagnosis_codes[0] = self.adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],
                                                                  torch.LongTensor([y] * t_diagnosis_codes[0].shape[
                                                                      0]).cuda()).detach()
                logit = self.model(t_diagnosis_codes[0], t_diagnosis_codes[1])
                logit = logit.data.cpu().numpy()
                subsets_g = logit[:, y]
                subsets_h = np.max([logit[:, false_class] for false_class in list_label_set], axis=0)
                subsets_object = subsets_h - subsets_g
                # get the maximum object, and update worst object
                temp_max_object = np.max(subsets_object)
                temp_max_index = np.argmax(subsets_object) + batch_size * index

                if temp_max_object > max_object:
                    max_object = temp_max_object
                    max_index = temp_max_index
            poses = np.array(list(pos_dict.keys()))
            max_pos_index = np.where(poses > max_index)[0][0]
            max_pos = pos_dict[poses[max_pos_index]]

            greedy_set_best_temp_funccall = funccall_lists_all[max_index]
            greedy_set_best_temp_funccall = np.array(greedy_set_best_temp_funccall)
            with ctx_noparamgrad_and_eval(self.model):
                t_diagnosis_codes = self.input_handle(greedy_set_best_temp_funccall)
                t_diagnosis_codes[0] = self.adversary.perturb(t_diagnosis_codes[0], t_diagnosis_codes[1],  # Perturb continuous features
                                                              torch.LongTensor([y]).cuda()).detach()
                greedy_set_best_temp_funccall[:self.n_con_fea] = t_diagnosis_codes[0].cpu().numpy()

            # print(iteration)
            # print('query', query_num)
            # print(max_object)

            greedy_set.add(max_pos)
            greedy_set_visit_idx.add(max_pos[0])
            pred, g, h = self.classify(greedy_set_best_temp_funccall, y)
            g_process.append(float(g))
            mf_process.append(float(h - g))
            greedy_set_process.append(copy.deepcopy(greedy_set))
            if max_object > current_object:
                current_object = max_object
            if max_object > 0:  #Target function >0, attack
                success_flag = 0
            changed_set_process.append(self.changed_set(funccall, greedy_set_best_temp_funccall))

            # print(greedy_set)
            if success_flag == 1:
                # if (time.time() - st) > time_limit or len(greedy_set) == num_feature[self.Dataset]:
                if iteration == self.budgets or len(greedy_set) == num_feature[self.Dataset]:
                    success_flag = -1
                    robust_flag = 1
                    # print('Time out')

        n_changed = len(self.changed_set(funccall, greedy_set_best_temp_funccall))

        if robust_flag == 0:
            flip_funccall = greedy_set_best_temp_funccall
            flip_set = self.changed_set(funccall, flip_funccall)
            # print('Attack successfully')
        #
        # print("Modified_set:", flip_set)
        # print(flip_funccall)

        return g_process, mf_process, greedy_set_process, changed_set_process, \
            query_num, robust_flag, greedy_set, greedy_set_visit_idx, \
            greedy_set_best_temp_funccall, \
            n_changed, flip_funccall, flip_set, iteration


# Set all random seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def invalid_sample_(Dataset): # Used to generate invalid sample template, its role is generate specific invalid feature fill pattern for different data types (multi-class/binary/mixed features)
    if Dataset_type[Dataset] == 'multi': # Each feature's invalid value filled with last category (eg [num_category-1, ..., num_category-1]), and do One-Hot encoding.
        invalid_sample = torch.tensor([num_category[Dataset] - 1] * num_feature[Dataset]).unsqueeze(0)
        invalid_sample = one_hot_samples(invalid_sample, Dataset)[0].cuda() # invalid_sample is (1, 60, 5), indexing [0] becomes (60, 5)
    elif Dataset_type[Dataset] == 'binary': # Invalid sample for binary features represented by all 0 vector
        invalid_sample = torch.tensor([0] * num_feature[Dataset]).cuda()
    elif Dataset_type[Dataset] == 'mixed': # Categorical part use last category, continuous part use 0
        invalid_sample_cat = torch.tensor([num_category[Dataset] - 1] * num_feature[Dataset]).unsqueeze(0)
        invalid_sample_cat = one_hot_samples(invalid_sample_cat, Dataset)[0].cuda()
        invalid_sample_con = torch.tensor([0] * num_con_feature[Dataset]).cuda()
        invalid_sample = [invalid_sample_con, invalid_sample_cat]
    else:
        invalid_sample = None
    return invalid_sample


def valid_mat_(Dataset, model): # Generate a valid matrix
    if Dataset in complex_categories.keys(): # If Dataset is member of complex categories, call model's valid_matrix method, generate valid matrix, move to GPU via .cuda()
        if isinstance(model, torch.nn.DataParallel):
            valid_mat = model.module.valid_matrix().cuda()  # Access original model via .module
        else:
            valid_mat = model.valid_matrix().cuda()  # Directly access original model
        # valid_mat = model.valid_matrix().cuda() # Model's valid_matrix method responsible for generating matrix adapted to current dataset
    else: # [feature count, category count]
        # num_feature = {'Splice': 60, 'pedec': 5000, 'census': 32} # Categorical feature count
        # num_category = {'Splice': 5, 'pedec': 3, 'census': 52}  # Splice dataset; each feature has 5 categories (possible values T, A, G, C, N)
        valid_mat = torch.ones(num_feature[Dataset], num_category[Dataset]).cuda() # valid_mat will be (num_feature[Dataset], num_category[Dataset]) all-1 matrix
        valid_mat[:, -1] = 0 # Set last column to 0
    return valid_mat