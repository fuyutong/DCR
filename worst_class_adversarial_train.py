import os
import time
import argparse
from functools import partial

import torchvision
import torch.optim as optim
from torchvision import transforms
from advertorch.attacks import L1PGDAttack
import datetime
from models import *
import numpy as np
import attack_generator as attack
from data_loader import get_cifar10_loader, get_cifar100_loader, get_multi_loader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.Training_utils import *

parser = argparse.ArgumentParser(description='PyTorch Worst-class Adversarial Training ')
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10, cifar100")
parser.add_argument('--net', type=str, default="MLP",help="decide which network to use,choose from resnet18,WRN")  # resnet18
parser.add_argument('--alg', type=str, default='none', help='use which algorithm')
parser.add_argument('--wat_bsl', type=lambda x: x.lower() == 'true', default=False, help='whether add BSL in WAT')
parser.add_argument('--wat_tail', type=lambda x: x.lower() == 'true', default=False, help='whether add TAIL in WAT')
parser.add_argument('--wat_rbl', type=lambda x: x.lower() == 'true', default=False, help='whether add RBL in WAT')
parser.add_argument('--wat_wct', type=lambda x: x.lower() == 'true', default=True, help='whether to use WCT dynamic class weights')
parser.add_argument('--wct_mode', type=str, default='cumsum',
                    choices=['cumsum', 'current', 'freq5'],
                    help='WCT weight update mode: cumsum=累积历史(default), current=仅当前epoch, freq5=每5epoch更新')
parser.add_argument('--weight_decay', '--wd', default=None, type=float, metavar='W')  # 2e-4 1e-4
parser.add_argument('--lr', type=float, default=0.0, metavar='LR', help='learning rate')
parser.add_argument('--idx', default=0, type=int, help='running index')
parser.add_argument('--batch_size', type=int, default=0, help='batch size for data_loader')
parser.add_argument('--epoch', type=int, default=0, metavar='N', help='number of epoch to train')

parser.add_argument('--eta', type=float, default=0.1, help='hyper-parameter used in WAT')
parser.add_argument('--beta', type=float, default=1, help='regularization parameter')  # 6.0
parser.add_argument('--beta_wat', type=float, default=1, help='regularization parameter')  # 6.0
parser.add_argument('--delta_ratio', type=float, default=0.001, help='delta ratio')
parser.add_argument('--epsilon', type=float, default=0.0, help='perturbation bound')  # 0.031 5
parser.add_argument('--num_steps', type=int, default=15, help='maximum perturbation step K')  # 10
parser.add_argument('--scale', type=float, default=1, help='Trades scale')  # 10  10e8
# WRN的参数
parser.add_argument('--depth', type=int, default=34, help='WRN depth')
parser.add_argument('--width_factor', type=int, default=10, help='WRN width factor')
parser.add_argument('--drop_rate', type=float, default=0.0, help='WRN drop rate')
parser.add_argument('--step_size', type=float, default=0.007, help='step size')
parser.add_argument('--momentum', type=float, default=0.1, metavar='M', help='SGD momentum')  # 0.9
# 不怎么变的参数
parser.add_argument('--resume', type=str, default='', help='whether to resume training')
parser.add_argument('--out_dir',type=str,default='./results/',help='dir of output')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed')
parser.add_argument('--rand_init', type=bool, default=True, help="whether to initialize adversarial sample with random noise")

args = parser.parse_args()
Dataset = args.dataset

lr = lr_list[Dataset] if args.lr == 0.0 else args.lr
n_epoch = epochs[Dataset] if args.epoch == 0 else args.epoch
batch_size = batch_sizes[Dataset] if args.batch_size == 0 else args.batch_size
class_num = num_classes[Dataset] # if args.class_num == 0 else args.class_num
epsilon = budgets[Dataset] if args.epsilon == 0.0 else args.epsilon
weight_decay = weight_decays[Dataset] if args.weight_decay is None else args.weight_decay

print('dataset:', Dataset, 'net', args.net, ',alg:', args.alg, ',wat_bsl:', args.wat_bsl,',wat_rbl:', args.wat_rbl,',wat_tail:', args.wat_tail,
      ',wat_wct:', args.wat_wct,',wct_mode:', args.wct_mode,',lr:', lr, ',weight_decay:', weight_decay,
      ',eta',args.eta, ',beta:', args.beta, ',beta_wat:', args.beta_wat, ',delta_rati', args.delta_ratio, ',epsilon', epsilon,
      ',num_steps:', args.num_steps, ',epoch:', n_epoch, ',batch_size:', batch_size, ',class_num:', class_num)
print("budgets[Dataset]:", budgets[Dataset])
# settings
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


print('==> Load Test Data')
if args.dataset == "cifar10":
    train_loader, valid_loader, test_loader = get_cifar10_loader(batch_size)
    class_num = 10
if args.dataset == "cifar100":
    train_loader, valid_loader, test_loader = get_cifar100_loader(batch_size)
    class_num = 100
if args.dataset in ["UC_multi","UC_multi_balanced","UC_multi_imbalanced","UC_multi_13","stroke_multi","Thyroid_multi","Thyroid_multi_balanced", "diatri_multi","cardio_multi","Splice"] :
    train_loader, valid_loader, test_loader, samples_per_cls, class_counts_dict = get_multi_loader(batch_size, Dataset)
    class_num = len(torch.unique(train_loader.dataset.labels))  # 自动获取类别数


overall = []
overall_error = []
pre_at_sample = copy.deepcopy(samples_per_cls)
next_at_sample = [0 for i in range(len(samples_per_cls))]
overall.append(samples_per_cls)
overall_error.append(samples_per_cls)
piror_p = [0 for i in range(len(samples_per_cls))]
post_p = copy.deepcopy(samples_per_cls)
print("original pre_at_sample:", pre_at_sample)


if args.dataset == 'UC_multi' or args.dataset == 'UC_multi_13' or args.dataset == 'UC_multi_balanced' or args.dataset == 'UC_multi_imbalanced':
    from models.UC_MultiModels import *
elif args.dataset == 'stroke_multi':
    from models.Stroke_MultiModels import *
elif args.dataset == 'Thyroid_multi' or args.dataset == 'Thyroid_multi_balanced' or args.dataset == 'diatri_multi':
    from models.Thyroid_MultiModels import *
elif args.dataset == 'Splice':
    from models.SpliceModels import *
elif args.dataset == 'cardio_multi':
    from models.Thyroid_MultiModels import *
else:
    raise NotImplementedError(f'Dataset not recognized ({args.dataset})')

print('==> Load Model')
if args.net == 'Trsf':
    model = Transformer(Dataset)
    net = 'Transformer'
elif args.net == "MLP":
    model = MLP(Dataset)
    net = "MLP"
elif args.net == "TabNet":
    model = TabNet(Dataset,
                   n_steps=tabnet_n_steps[Dataset],
                   n_d=tabnet_n_d[Dataset],
                   n_a=tabnet_n_a[Dataset],
                   gamma=tabnet_gamma[Dataset])
    net = "TabNet"

nat_class_weights = torch.ones(class_num+1).cuda()
bndy_class_weights = torch.ones(class_num+1).cuda()

model = torch.nn.DataParallel(model)
print(model)
print(net)
print("CUDA available:", torch.cuda.is_available())  # 是否检测到 GPU
print("Device count:", torch.cuda.device_count())    # 可用 GPU 数量
print("Current device:", torch.cuda.current_device())# 当前使用 GPU 索引
print("Device name:", torch.cuda.get_device_name(0)) # GPU 型号名称

# 在 model 定义之后、optimizer 定义之前加入
if args.net == 'TabNet' and Dataset in ['UC_multi', 'UC_multi_balanced', 'UC_multi_imbalanced']:
    lr = lr * 0.3   # TabNet 在小数据集上需要更小的学习率

def TRADES_loss(adv_logits, natural_logits, target, beta):
    # Based on the repo of TREADES: https://github.com/yaodongyu/TRADES

    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()  # 新版替换如果 size_average=True → 对应 reduction='mean'。# 如果 size_average=False → 对应 reduction='sum'。# 如果 reduce=False → 对应 reduction='none'。
    # criterion_kl = nn.KLDivLoss(reduction='sum').cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits + 1e-10, dim=1), F.softmax(natural_logits + 1e-10, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, loss_natural, loss_robust


def TRADES_classwise_loss(adv_logits, natural_logits, t_diagnosis_codes, output_adv, target, sample_pre_class, at_pre_class):

    if args.wat_bsl:
        for i in range(len(sample_pre_class)):
            if sample_pre_class[i] <= 0:
                sample_pre_class[i] = 1
        spc = torch.tensor(sample_pre_class).type_as(t_diagnosis_codes)
        spc = spc.unsqueeze(0).expand(t_diagnosis_codes.shape[0], -1)

        if args.wat_rbl:
            beta = np.zeros(len(sample_pre_class)).astype(np.float32)
            E = np.zeros(len(sample_pre_class)).astype(np.float32)
            for i in range(len(sample_pre_class)):
                beta[i] = (sample_pre_class[i] - 1.) / sample_pre_class[i]
                E[i] = (1. - beta[i] ** at_pre_class[i]) / (1. - beta[i])
            weights = 1. / (E + 1e-5)
            weights = weights / np.sum(weights) * len(sample_pre_class)
            loss_natural = nn.CrossEntropyLoss(reduction='none', weight=torch.from_numpy(weights.astype(np.float32)).cuda())(natural_logits + spc.log(), target)
        else:
            loss_natural = nn.CrossEntropyLoss(reduction='none')(natural_logits + spc.log(), target)

        criterion_kl = nn.KLDivLoss(reduction='none').cuda()
        loss_robust = criterion_kl(F.log_softmax(adv_logits + 1e-10 + spc.log(), dim=1), F.softmax(natural_logits + 1e-10 + spc.log(), dim=1))
        loss_robust_sum = loss_robust.sum(dim=1)
    else:
        loss_natural = nn.CrossEntropyLoss(reduction='none')(natural_logits, target)
        criterion_kl = nn.KLDivLoss(reduction='none').cuda()
        loss_robust = criterion_kl(F.log_softmax(adv_logits + 1e-10, dim=1), F.softmax(natural_logits + 1e-10, dim=1))
        loss_robust_sum = loss_robust.sum(dim=1)
    if args.wat_tail:
        kl = nn.KLDivLoss(size_average='none').cuda()  # 算出来的kl_的维度是tensor()  旧版 size_average='none' 实际上等同于 reduction='sum'（默认对所有元素求和）

        spc = torch.tensor(samples_per_cls).type_as(t_diagnosis_codes)
        weights = torch.sqrt(1. / (spc / spc.sum()))  # 数量越少，权重越大 ，使用平方根进行平滑。

        # 按样本数量从小到大排序类别索引
        sorted_classes = sorted(class_counts_dict.keys(), key=lambda x: class_counts_dict[x])  # 按数量对类别索引排序
        # 动态计算尾部类别数量（至少1个）
        num_tail = max(1, len(sorted_classes) // 3)
        tail_class = sorted_classes[:num_tail]  # 样本最少的num_tail个类别

        # calculate robust loss
        f_adv, logits = model(output_adv, True)  # 特征空间正则化：通过约束对抗样本的特征分布（f_adv），使模型在特征空间中对不同类别（尤其是尾部类别）的分布更加鲁棒。

        TAIL = 0.0
        counter = 0.0
        for bi in range(target.size(0)):
            if target[bi].item() in tail_class:  # 第二个样本是0，在少数类里面
                idt = torch.tensor(  # 创建身份张量：如果当前样本与批次中的其他样本属于同一类别，则为-1；否则为1。这样可以使同类样本特征更相似（KL散度更小），不同类样本特征更不同（KL散度更大）。
                    [-1. if target[bi].item() == target[bj].item() else 1. for bj in range(target.size(0))]).cuda()
                W = torch.tensor(
                    [weights[target[bi].item()] + weights[target[bj].item()] for bj in  # 给不是自己类别的样本和是自己类别的样本分别加权
                     range(target.size(0))]).cuda()  # 当前样本类别的权重加上批次中每个样本类别的权重。这样稀有类别之间的交互会获得更高的权重。

                kl_ = kl(F.log_softmax(f_adv + 1e-10, 1), F.softmax(
                    f_adv[bi, :].clone().detach().view(1, -1).tile(target.size(0), ).view(target.size(0), -1),
                    1))  # 计算当前样本的特征与批次中所有样本特征之间的KL散度，然后乘以身份张量和权重。

                TAIL += kl_ * idt * W  # 得了我还是用以前的吧，把warning 给ignore一下
                counter += 1  # 增加计数器，记录处理的尾部类别样本数量。
        TAIL = TAIL.mean() / counter if counter > 0. else 0.0  # 计算平均TAIL损失：先对所有维度求平均，再除以处理的样本数量。如果没有处理任何样本，则损失为0。
        loss_robust_sum += TAIL


    # scale = 10e8
    return loss_natural, args.scale * loss_robust_sum  # 每个样本在两个类别上的kl散度的和  abs()


def BSL(labels, logits, sample_per_class):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
	Args:
	  labels: A int tensor of size [batch].
	  logits: A float tensor of size [batch, no_of_classes].
	  sample_per_class: A int tensor of size [no of classes].
	  reduction: string. One of "none", "mean", "sum"
	Returns:
	  loss: A float tensor. Balanced Softmax Loss.
	"""
    for i in range(len(sample_per_class)):
        if sample_per_class[i] <= 0:
            sample_per_class[i] = 1
    spc = torch.tensor(sample_per_class).type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels)
    return loss


def RBL(logits, labels, sample_pre_class, at_pre_class):
    beta = np.zeros(len(sample_pre_class)).astype(np.float32)
    E = np.zeros(len(sample_pre_class)).astype(np.float32)
    for i in range(len(sample_pre_class)):
        beta[i] = (sample_pre_class[i] - 1.) / sample_pre_class[i]
        E[i] = (1. - beta[i]**at_pre_class[i]) / (1. - beta[i])
    weights = 1. / (E + 1e-5)
    weights = weights / np.sum(weights) * len(sample_pre_class)
    loss = F.cross_entropy(logits, labels, weight=torch.from_numpy(weights.astype(np.float32)).cuda())
    return loss


class RBLAttackWrapper:
    def __init__(self, model,  samples_per_cls, at_pre_class, eps=0.1, nb_iter=20, eps_iter=0.01):
        self.model = model
        self.samples_per_cls = samples_per_cls
        self.at_pre_class = at_pre_class
        # 初始化攻击器（延迟创建以便动态更新参数）
        self.adversary = None
        self._init_adversary(eps, nb_iter, eps_iter)

    def _init_adversary(self, eps, nb_iter, eps_iter):
        # 动态绑定当前实例的方法作为损失函数
        self.adversary = L1PGDAttack(
            self.model,
            loss_fn=self.rbl_loss,  # 使用绑定方法
            eps=eps,
            nb_iter=nb_iter,
            eps_iter=eps_iter,
            rand_init=True,
            clip_min=0.0,
            clip_max=1.0,
            targeted=False
        )
    def rbl_loss(self, logits_adv, y):
        # 强制模型输出 logits
        # with torch.no_grad():
        #     _, logits_adv = self.model(x_adv, True)
        # 调用 RBL 计算损失
        return RBL(logits=logits_adv, labels=y, sample_pre_class=self.samples_per_cls, at_pre_class=self.at_pre_class)

    def perturb(self, x, y):
        # 生成对抗样本
        with ctx_noparamgrad_and_eval(self.model):
            adv_data = self.adversary.perturb(x, y).detach()
        valid_mat = valid_mat_(Dataset, model)
        output_adv = adv_data * valid_mat
        return output_adv


def REAT(model, x, y, optimizer, sample_per_class, at_per_class):
    model.eval()
    epsilon = args.eps
    num_steps = args.ns
    step_size = args.ss
    x_adv = x.detach() + torch.FloatTensor(*x.shape).uniform_(-epsilon, epsilon).cuda()
    for _ in range(num_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            f_adv, logits_adv = model(x_adv, True)
            loss = RBL(y, logits_adv, sample_per_class, at_per_class)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)  # 旧pytorch1.8的写法
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()  # 新pytorch2.3.0的写法
    rbl_loss_partial = partial(RBL, sample_pre_class=samples_per_cls)  # , at_pre_class=pre_at_sample 是动态的，不是固定的，所以要放在外面
    adversary = L1PGDAttack(model, loss_fn=rbl_loss_partial, eps=float(epsilon),
                            nb_iter=20, eps_iter=float(epsilon) / 10, rand_init=True, clip_min=0.0,
                            clip_max=1.0, targeted=False)
    with ctx_noparamgrad_and_eval(model):
        adv_data = adversary.perturb(x, y).detach()
    valid_mat = valid_mat_(Dataset, model)
    x_adv = adv_data * valid_mat
    model.train()
    # zero gradient
    optimizer.zero_grad()
    return x_adv


def pgdBSL_loss(model,
             x,
             y,
             samples_per_cls,
             optimizer):
    model.eval()
    # generate adversarial example
    with ctx_noparamgrad_and_eval(model):
        adv_data = adversary.perturb(x, y).detach()
    valid_mat = valid_mat_(Dataset, model)
    x_adv = adv_data * valid_mat

    model.train()
    # zero gradient
    optimizer.zero_grad()

    loss = BSL(y, model(x_adv), samples_per_cls)
    return loss

import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, epoch, log_train):
    global iter_num
    global nat_class_weights
    global bndy_class_weights

    starttime = datetime.datetime.now()
    loss_sum = 0
    loss_nat_sum = 0
    loss_rob_sum = 0
    loss_wat_sum = 0
    loss_bsl_sum =0
    loss_tail_sum = 0
    train_iter = 0
    correct = 0
    # 新增：收集所有样本的真实标签和预测概率（用于 AUC 计算）
    y_true = []
    y_prob = []
    all_adv_features = []  # 存储所有对抗样本的特征
    all_latent_adv_features = []
    all_labels = []  # 存储所有样本的标签

    for batch_diagnosis_codes, target in train_loader:
        train_iter+=1
        iter_num+=1
        # data, target = data.cuda(), target.cuda()
        target = target.cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

        if args.alg == 'Normal':
            natural_logits = model(t_diagnosis_codes)
            latent_f_adv, _ = model(t_diagnosis_codes, True)
            loss = CEloss(natural_logits, target)
            # --- 新增逻辑：收集真实标签和预测概率 ---
            # 将标签和概率转换为 CPU 上的 numpy 数组
            y_true.extend(target.cpu().numpy())  # 真实标签 (shape: [batch_size])
            y_prob.extend(
                F.softmax(natural_logits + 1e-10, dim=1).detach().cpu().numpy())  # 预测概率 (shape: [batch_size, num_classes])
            pred = natural_logits.max(1, keepdim=True)[1]
            target = target.view_as(pred)
            target_view = target.view_as(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1 == 0:
                for j in range(pred.size(0)):
                    next_at_sample[pred[j].item()] += 1  # 每个类别被 对抗攻击 预测为类别的次数总和
                    if pred[j].item() != target_view[j].item():
                        piror_p[target_view[j].item()] += 1  # 对于每个 真实类别，对抗攻击导致预测错误的次数

            # --- 新增逻辑：收集真实标签和预测概率 ---
            # 将标签和概率转换为 CPU 上的 numpy 数组
            all_adv_features.append(natural_logits.detach().cpu().numpy())  # 对抗样本特征
            all_latent_adv_features.append(latent_f_adv.detach().cpu().numpy())
            all_labels.append(target.cpu().numpy())  # 将标签转移到 CPU 并转换为 numpy 数组
            continue
        # 对抗训练情况，生成对抗样本部分
        if 'PGD' in args.alg:
            with ctx_noparamgrad_and_eval(model):
                adv_data = adversary.perturb(t_diagnosis_codes, target).detach()
            valid_mat = valid_mat_(Dataset, model)
            output_adv = adv_data * valid_mat
        elif 'REAT' in args.alg or 'RBL' in args.alg or args.wat_rbl:
            # output_adv = REAT(model, t_diagnosis_codes, target, optimizer, samples_per_cls, pre_at_sample)
            rbl_attacker = RBLAttackWrapper(model=model, samples_per_cls=samples_per_cls, at_pre_class=pre_at_sample, eps=float(epsilon), nb_iter=20, eps_iter=float(epsilon) / 10)
            output_adv = rbl_attacker.perturb(t_diagnosis_codes, target)
        else:
            output_adv = attack.trades(model,t_diagnosis_codes,target, epsilon=epsilon, step_size=args.step_size, num_steps=args.num_steps,
                                        loss_fn='trades', category='trades', rand_init=True, Dataset=Dataset, delta_ratio=args.delta_ratio)
        # set model.train after be attacked
        model.train()
        optimizer.zero_grad()

        natural_logits = model(t_diagnosis_codes)
        adv_logits = model(output_adv)
        latent_f_adv,_= model(output_adv, True)   # ,latent_f_adv2,latent_f_adv3
        # --- 新增逻辑：收集真实标签和预测概率 ---
        # 将标签和概率转换为 CPU 上的 numpy 数组
        all_adv_features.append(adv_logits.detach().cpu().numpy())  # 对抗样本特征
        all_latent_adv_features.append(latent_f_adv.detach().cpu().numpy())
        all_labels.append(target.cpu().numpy())  # 将标签转移到 CPU 并转换为 numpy 数组

        y_true.extend(target.cpu().numpy())  # 真实标签 (shape: [batch_size])
        y_prob.extend(F.softmax(natural_logits, dim=1).detach().cpu().numpy())  # 预测概率 (shape: [batch_size, num_classes])

        # 开始算Loss
        if 'old' in args.alg:
            # 替换之前的 作者原来的代码
            iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits, natural_logits, t_diagnosis_codes, output_adv, target, sample_pre_class=samples_per_cls, at_pre_class=pre_at_sample)

            for i in range(class_num):
                if i == 0:
                    nat_loss = iter_nat_loss[target == i].sum() * nat_class_weights[i]  # 获取target == i的索引加和
                    bndy_loss = iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
                else:
                    nat_loss += iter_nat_loss[target == i].sum() * nat_class_weights[i]
                    bndy_loss += iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
            wat_loss = (nat_loss + args.beta * bndy_loss) / t_diagnosis_codes.shape[0]
            loss = wat_loss
            loss_wat_sum += wat_loss.item()
            # In WAT, we add excepted training loss in the decision set.
            if 'wat' in args.alg:  # our method
                loss0, loss_nat, loss_rob = TRADES_loss(adv_logits, natural_logits, target, args.beta)
                loss += loss0 * bndy_class_weights[class_num]
                loss_nat_sum += loss_nat.item()
                loss_rob_sum += loss_rob.item()
        elif 'new' in args.alg:
            loss0, loss_nat, loss_rob = TRADES_loss(adv_logits, natural_logits, target, args.beta)
            loss = loss0 * bndy_class_weights[class_num]
            loss_nat_sum += loss_nat.item()
            loss_rob_sum += loss_rob.item()
            # In WAT, we add excepted training loss in the decision set.
            if 'wat' in args.alg: # our method
                iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits, natural_logits, t_diagnosis_codes, output_adv, target, sample_pre_class=samples_per_cls, at_pre_class=pre_at_sample)

                for i in range(class_num):
                    if i == 0:
                        nat_loss = iter_nat_loss[target == i].sum() * nat_class_weights[i]  # 获取target == i的索引加和
                        bndy_loss = iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
                    else:
                        nat_loss += iter_nat_loss[target == i].sum() * nat_class_weights[i]
                        bndy_loss += iter_bndy_loss[target == i].sum() * bndy_class_weights[i]
                wat_loss = (nat_loss + args.beta * bndy_loss) / t_diagnosis_codes.shape[0]
                loss += wat_loss
                loss_wat_sum += wat_loss.item()
        # else:
        #     loss, loss0, loss_nat, loss_rob = 0, 0, 0, 0 # Normal_classwise_loss(adv_logits, natural_logits, target)
        else:
            if 'BSL' in args.alg:
                BSL_loss = BSL(target, adv_logits, samples_per_cls)
                # BSL_loss = pgdBSL_loss(model, t_diagnosis_codes, target, samples_per_cls, optimizer)
                loss = BSL_loss
                loss_bsl_sum += BSL_loss.item()
            if 'REAT' in args.alg or 'RBL' in args.alg:
                if 'BSL' in args.alg:  # （2）RBL + BSL模式
                    loss = loss
                else:
                    kl = nn.KLDivLoss(size_average='none').cuda()

                    spc = torch.tensor(samples_per_cls).type_as(t_diagnosis_codes)
                    weights = torch.sqrt(1. / (spc / spc.sum()))

                    # 按样本数量从小到大排序类别索引
                    sorted_classes = sorted(class_counts_dict.keys(), key=lambda x: class_counts_dict[x])  # 按数量对类别索引排序
                    # 动态计算尾部类别数量（至少1个）
                    num_tail = max(1, len(sorted_classes) // 3)
                    tail_class = sorted_classes[:num_tail]  # 样本最少的num_tail个类别

                    # calculate robust loss
                    f_adv, logits = model(output_adv, True)  # 特征空间正则化：通过约束对抗样本的特征分布（f_adv），使模型在特征空间中对不同类别（尤其是尾部类别）的分布更加鲁棒。

                    TAIL = 0.0
                    counter = 0.0
                    for bi in range(target.size(0)):
                        if target[bi].item() in tail_class:
                            idt = torch.tensor([-1. if target[bi].item() == target[bj].item() else 1. for bj in range(target.size(0))]).cuda()
                            W = torch.tensor([weights[target[bi].item()] + weights[target[bj].item()] for bj in range(target.size(0))]).cuda()
                            kl_ = kl(F.log_softmax(f_adv, 1), F.softmax(
                                f_adv[bi, :].clone().detach().view(1, -1).tile(target.size(0), ).view(target.size(0), -1), 1))
                            TAIL += kl_ * idt * W
                            counter += 1
                    TAIL = TAIL.mean() / counter if counter > 0. else torch.tensor(0.0, requires_grad=True, device='cuda') # 0.0
                    if args.alg == 'REAT':   # （1）REAT完全版模式 包括 RBL生成对抗样本 + BSL + TAIL
                        loss = BSL(target, logits, samples_per_cls) + TAIL
                        loss_bsl_sum += BSL(target, logits, samples_per_cls).item()
                    else:  # （3）RBL + TAIL模式
                        loss = TAIL
                    loss_tail_sum += TAIL
            elif 'TAIL' in args.alg:  # （4） 单独的TAIL组件，一般用在PGDBSL + TAIL中
                kl = nn.KLDivLoss(size_average='none').cuda() # 算出来的kl_的维度是tensor()  旧版 size_average='none' 实际上等同于 reduction='sum'（默认对所有元素求和）

                spc = torch.tensor(samples_per_cls).type_as(t_diagnosis_codes)
                weights = torch.sqrt(1. / (spc / spc.sum()))  # 数量越少，权重越大 ，使用平方根进行平滑。

                # 按样本数量从小到大排序类别索引
                sorted_classes = sorted(class_counts_dict.keys(), key=lambda x: class_counts_dict[x])  # 按数量对类别索引排序
                # 动态计算尾部类别数量（至少1个）
                num_tail = max(1, len(sorted_classes) // 3)
                tail_class = sorted_classes[:num_tail]  # 样本最少的num_tail个类别

                # calculate robust loss
                f_adv, logits = model(output_adv, True)  # 特征空间正则化：通过约束对抗样本的特征分布（f_adv），使模型在特征空间中对不同类别（尤其是尾部类别）的分布更加鲁棒。

                TAIL = 0.0
                counter = 0.0
                for bi in range(target.size(0)):
                    if target[bi].item() in tail_class:  # 第二个样本是0，在少数类里面
                        idt = torch.tensor(  # 创建身份张量：如果当前样本与批次中的其他样本属于同一类别，则为-1；否则为1。这样可以使同类样本特征更相似（KL散度更小），不同类样本特征更不同（KL散度更大）。
                            [-1. if target[bi].item() == target[bj].item() else 1. for bj in range(target.size(0))]).cuda()
                        W = torch.tensor([weights[target[bi].item()] + weights[target[bj].item()] for bj in   # 给不是自己类别的样本和是自己类别的样本分别加权
                                          range(target.size(0))]).cuda()  # 当前样本类别的权重加上批次中每个样本类别的权重。这样稀有类别之间的交互会获得更高的权重。

                        kl_ = kl(F.log_softmax(f_adv, 1), F.softmax(f_adv[bi, :].clone().detach().view(1, -1).tile(target.size(0), ).view(target.size(0), -1), 1))  # 计算当前样本的特征与批次中所有样本特征之间的KL散度，然后乘以身份张量和权重。

                        TAIL += kl_ * idt * W  # 得了我还是用以前的吧，把warning 给ignore一下
                        counter += 1  # 增加计数器，记录处理的尾部类别样本数量。
                TAIL = TAIL.mean() / counter if counter > 0. else torch.tensor(0.0, requires_grad=True, device='cuda')  # 0.0  # 计算平均TAIL损失：先对所有维度求平均，再除以处理的样本数量。如果没有处理任何样本，则损失为0。
                loss += TAIL
                loss_tail_sum += TAIL  # .item()
            else:
                loss = CEloss(adv_logits, target) + CEloss(natural_logits, target)

        loss_sum += loss.item() if type(loss) == torch.Tensor else loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = adv_logits.max(1, keepdim=True)[1]
        target_view = target.view_as(pred)

        if (epoch + 1) % 1 == 0:
            for j in range(pred.size(0)):
                next_at_sample[pred[j].item()] += 1  # 每个类别被 对抗攻击 预测为类别的次数总和
                if pred[j].item() != target_view[j].item():
                    piror_p[target_view[j].item()] += 1  # 对于每个 真实类别，对抗攻击导致预测错误的次数
    # --- 计算 AUC ---
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    all_adv_features = np.concatenate(all_adv_features, axis=0)  # 拼接所有对抗样本特征
    all_latent_adv_features = np.concatenate(all_latent_adv_features, axis=0)  # 拼接所有对抗样本特征
    all_labels = np.concatenate(all_labels, axis=0)  # 拼接所有样本标签
    np.save(args.out_dir + f'/npyresult/{args.net}_{args.alg}_{batch_size}/{args.idx}_epoch_{epoch}_adv_features.npy', all_adv_features)
    np.save(args.out_dir + f'/npyresult/{args.net}_{args.alg}_{batch_size}/{args.idx}_epoch_{epoch}_latent_adv_features.npy', all_latent_adv_features)
    np.save(args.out_dir + f'/npyresult/{args.net}_{args.alg}_{batch_size}/{args.idx}_epoch_{epoch}_labels.npy', all_labels)

    train_clean_accuracy = correct / len(train_loader.dataset)
    writer.add_scalar('acc/train_clean_accuracy', train_clean_accuracy, epoch + 1)
    auc_score = compute_auc(y_true, y_prob, Dataset)

    endtime = datetime.datetime.now()
    time = (endtime - starttime).seconds
    print('Epoch: [%d | %d] | Train Time: %.5f s | Loss %f | Nat Loss %f | Rob Loss %f | WAT Loss %f | BSL Loss %f | TAIL Loss %f | Train AUC %f' % (
        epoch + 1, n_epoch, time, loss_sum, loss_nat_sum, loss_rob_sum, loss_wat_sum, loss_bsl_sum, loss_tail_sum, auc_score))
    print('Epoch: [%d | %d] | Train Time: %.5f s | Loss %f | Nat Loss %f | Rob Loss %f | WAT Loss %f | BSL Loss %f | TAIL Loss %f | Train AUC %f' % (
            epoch + 1, n_epoch, time, loss_sum, loss_nat_sum, loss_rob_sum, loss_wat_sum, loss_bsl_sum, loss_tail_sum, auc_score), file=log_train, flush=True)

    return time, loss_sum


def validate_Normal(model, valid_loader):
    model.eval()
    val_loss = 0
    correct = 0
    class_wise_correct = []
    class_wise_num = []

    # 新增：收集所有样本的真实标签和预测概率（用于 AUC 计算）
    y_true = []
    y_prob = []
    for i in range(class_num):
        class_wise_correct.append(0)
        class_wise_num.append(0)
    with torch.no_grad():
        for batch_diagnosis_codes, target in test_loader:
            # data, target = data.cuda(), target.cuda()
            target = target.cuda()
            t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
            output = model(t_diagnosis_codes)

            val_loss += nn.CrossEntropyLoss(reduction='mean')(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            target = target.view_as(pred)
            eq_mat = pred.eq(target)
            correct += pred.eq(target.view_as(pred)).sum().item()
            for i in range(class_num):
                class_wise_num[i] += (target == i).int().sum().item()
                class_wise_correct[i] += eq_mat[target == i].sum().item()

            # --- 新增逻辑：收集真实标签和预测概率 ---
            # 将标签和概率转换为 CPU 上的 numpy 数组
            y_true.extend(target.cpu().numpy())  # 真实标签 (shape: [batch_size])
            y_prob.extend(F.softmax(output + 1e-10, dim=1).detach().cpu().numpy())  # 预测概率 (shape: [batch_size, num_classes])

    # --- 计算 AUC ---
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc_score = compute_auc(y_true, y_prob, Dataset)

    val_loss /= len(test_loader.dataset)
    val_accuracy = correct / len(test_loader.dataset)

    for i in range(class_num):
        class_wise_correct[i] /= class_wise_num[i]
    print(
        f'Val_Loss: {val_loss:.5f} | Val_Acc: {val_accuracy:.5f} | Val_class_wise_acc: {class_wise_correct} | Val_AUC: {auc_score:.5f} |')
    return val_loss, val_accuracy, class_wise_correct, auc_score


def validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost):
    val_iter = 0
    model.eval()
    epoch_val_bndy_loss = torch.zeros(class_num+1).cuda()
    epoch_val_bndy_loss.requires_grad = False
    epoch_val_nat_loss = torch.zeros(class_num+1).cuda()
    epoch_val_nat_loss.requires_grad = False
    correct = 0
    val_class_wise_acc = []
    val_class_wise_num = []
    for i in range(class_num):   # 给每个类别初始化acc和num
        val_class_wise_acc.append(0)
        val_class_wise_num.append(0)
    model.zero_grad()
    loss_cl_sum = 0
    loss_adv_sum = 0
    # 新增：收集所有样本的真实标签和预测概率（用于 AUC 计算）
    y_true = []
    y_prob = []
    for batch_diagnosis_codes, target in valid_loader:
        val_iter+=1
        # data, target = data.cuda(), target.cuda()
        target = target.cuda()
        t_diagnosis_codes = input_process(batch_diagnosis_codes, Dataset)
        t_diagnosis_codes = torch.autograd.Variable(t_diagnosis_codes.data, requires_grad=True)

        # 生成对抗样本部分
        if 'PGD' in args.alg:
            with ctx_noparamgrad_and_eval(model):
                adv_data = adversary.perturb(t_diagnosis_codes, target).detach()
            valid_mat = valid_mat_(Dataset, model)
            output_adv = adv_data * valid_mat
        # elif 'REAT' in args.alg:
        #     output_adv = REAT(model, t_diagnosis_codes, target, optimizer, samples_per_cls, pre_at_sample)
        elif 'REAT' in args.alg or 'RBL' in args.alg or args.wat_rbl:
            # output_adv = REAT(model, t_diagnosis_codes, target, optimizer, samples_per_cls, pre_at_sample)
            rbl_attacker = RBLAttackWrapper(model=model, samples_per_cls=samples_per_cls, at_pre_class=pre_at_sample,
                                            eps=float(epsilon), nb_iter=20, eps_iter=float(epsilon) / 10)
            output_adv = rbl_attacker.perturb(t_diagnosis_codes, target)
        else:
            output_adv = attack.trades(model, t_diagnosis_codes, target, epsilon=epsilon, step_size=args.step_size,
                                       num_steps=args.num_steps,
                                       loss_fn='trades', category='trades', rand_init=True, Dataset=Dataset,
                                       delta_ratio=args.delta_ratio)
            # output_adv = attack.trades(model, t_diagnosis_codes, target, epsilon, args.step_size, args.num_steps,"trades","trades",True, Dataset, args.delta_ratio)

        with torch.no_grad():
            natural_logits = model(t_diagnosis_codes)
            adv_logits = model(output_adv)

            pred = adv_logits.max(1, keepdim=True)[1]
            target_view = target.view_as(pred)
            eq_mask = pred.eq(target_view) # equal mask
            correct += pred.eq(target.view_as(pred)).sum().item()  # 不分类别，两个类别里都预测正确的数量
            for i in range(class_num):
                label_mask = target_view == i  # 这个类别下预测分类正确的bool
                val_class_wise_num[i] += label_mask.sum().item()  # 这个类别下正确的数量
                val_class_wise_acc[i] += (eq_mask * label_mask).sum().item()  # 分类正确的数量
            if 'wat' in args.alg or 'none' in args.alg:
                # class-wise
                iter_nat_loss, iter_bndy_loss = TRADES_classwise_loss(adv_logits, natural_logits, t_diagnosis_codes, output_adv, target, sample_pre_class=samples_per_cls, at_pre_class=pre_at_sample)
                for i in range(class_num):
                    epoch_val_nat_loss[i] += iter_nat_loss[target == i].sum()  # 属于这个类别下的nat_loss的和
                    epoch_val_bndy_loss[i] += iter_bndy_loss[target == i].sum()  # 属于这个类别下的bndy_loss的和
                epoch_val_nat_loss[class_num] += iter_nat_loss.sum() / class_num  # 如果只有2类，那这个是第3类，标记为所有总的nat_loss的和，放在了第三类
                epoch_val_bndy_loss[class_num] += iter_bndy_loss.sum() / class_num


            else:
                loss_cl = nn.CrossEntropyLoss(reduction='none')(natural_logits, target)
                loss_adv = nn.CrossEntropyLoss(reduction='mean')(adv_logits, target)
                loss_cl_sum += loss_cl.sum().item()
                loss_adv_sum += loss_adv.item()


        # --- 新增逻辑：收集真实标签和预测概率 ---
        # 将标签和概率转换为 CPU 上的 numpy 数组
        y_true.extend(target.cpu().numpy())  # 真实标签 (shape: [batch_size])
        y_prob.extend(
            F.softmax(natural_logits, dim=1).detach().cpu().numpy())  # 预测概率 (shape: [batch_size, num_classes])

    # --- 计算 AUC ---
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    auc_score = compute_auc(y_true, y_prob, Dataset)
    if 'wat' in args.alg or 'none' in args.alg:
        wat_valid_nat_cost[epoch] = epoch_val_nat_loss / (val_iter * batch_size) * class_num
        wat_valid_bndy_cost[epoch] = epoch_val_bndy_loss / (val_iter * batch_size) * class_num

    val_acc = correct / len(valid_loader.dataset)
    for i in range(class_num):
        val_class_wise_acc[i] /= val_class_wise_num[i]
    model.zero_grad()
    if 'wat' in args.alg or 'none' in args.alg:
        print(f'Val_nat_Loss: {wat_valid_nat_cost[epoch].sum().item():.8f} | Val_bndy_Loss: {wat_valid_bndy_cost[epoch].sum().item():.8f} | Val_Acc: {val_acc:.8f} | Val_class_wise_acc: {val_class_wise_acc} | Val_AUC: {auc_score:.8f} |')
        return wat_valid_nat_cost, wat_valid_bndy_cost, val_acc, val_class_wise_acc, auc_score
    else:
        print(f'Val_adv_Loss: {loss_adv_sum:.8f} | Val_clean_Loss: {loss_cl_sum:.8f} | Val_Acc: {val_acc:.8f} | Val_class_wise_acc: {val_class_wise_acc} | Val_AUC: {auc_score:.8f} |')
        return loss_adv_sum, loss_cl_sum, val_acc, val_class_wise_acc, auc_score


def adjust_learning_rate(lr0, optimizer, epoch):
    """decrease the learning rate"""
    lr = lr0
    # The same as TRADES used in CIFAR-10
    if epoch >= 60:
        lr = lr0 * 0.1
    if epoch >= 85:
        lr = lr0 * 0.01
    if epoch >= 100:
        lr = lr0 * 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint=args.out_dir, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

start_epoch = 0
args.out_dir = args.out_dir + '{}/'.format(Dataset)    #print(args.out_dir)
if not os.path.exists(args.out_dir+'npyresult/'):
    os.makedirs(args.out_dir+'npyresult/')
if not os.path.exists(args.out_dir+f'npyresult/{args.net}_{args.alg}_{batch_size}/'):
    os.makedirs(args.out_dir+f'npyresult/{args.net}_{args.alg}_{batch_size}/')
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
np.save(args.out_dir + 'npyresult/samples_per_cls.npy', np.array(samples_per_cls))  # 保存为 numpy 数组
print("successfully saved samples_per_cls.npy")

param_time = datetime.datetime.now().strftime("%I%M%S%p-%b%d")
print(param_time)

param_list = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(Dataset, args.net, args.alg,
                                                                '{}{}{}{}-{}'.format(
                                                                    'T' if args.wat_bsl else 'F',
                                                                    'T' if args.wat_tail else 'F',
                                                                    'T' if args.wat_rbl else 'F',
                                                                    'T' if args.wat_wct else 'F',
                                                                    args.wct_mode if args.wat_wct else 'none'
                                                                ), lr, n_epoch, args.eta,
                                                       args.beta, args.beta_wat, args.delta_ratio, epsilon, args.num_steps, args.scale, param_time)
if not os.path.exists(args.out_dir +'ckpt/'+ param_list):
    os.makedirs(args.out_dir +'ckpt/'+ param_list)
print(param_list)
iter_num = 0

if args.resume:
    print ('==> Adversarial Training Resuming from checkpoint ..')
    print(args.resume)
    assert os.path.isfile(args.resume)
    # args.out_dir = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'wat' in args.alg or 'none' in args.alg:
        wat_valid_nat_cost = checkpoint['wat_valid_nat_cost']
        wat_valid_bndy_cost = checkpoint['wat_valid_bndy_cost']
        wat_valid_cost = checkpoint['wat_valid_cost']
        best_val_nat_loss = checkpoint['val_nat_loss']
        best_val_bndy_loss = checkpoint['val_bndy_loss']
else:
    print('==> Worst-class Adversarial Training')

wat_valid_nat_cost = torch.zeros(n_epoch, class_num+1).cuda()
wat_valid_bndy_cost = torch.zeros(n_epoch, class_num+1).cuda()
wat_valid_cost = torch.zeros(n_epoch, class_num+1).cuda()

CEloss = torch.nn.CrossEntropyLoss().cuda()

if not os.path.exists(args.out_dir + 'Logs/Training/'):
    os.makedirs(args.out_dir + 'Logs/Training/')
# log_attack = open(f'./Logs/{Dataset}/training/attack_utils/Attack_{Model_Name}_Adam_{lr}.bak', 'w+')
log_train_test = open(args.out_dir + f'Logs/Training/Train_{param_list}.bak', 'w+')  # 'w+'是覆写模式 # ./resutls/{args.alg}/Logs/attack_utils
writer = SummaryWriter(log_dir=args.out_dir +'/visual/{}'.format(param_list))

if 'PGD' in args.alg:
    adversary = L1PGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=float(epsilon),
                            nb_iter=20, eps_iter=float(epsilon) / 10, rand_init=True, clip_min=0.0,
                            clip_max=1.0, targeted=False)

best_val_nat_loss = float('inf')
best_val_bndy_loss = float('inf')
best_val_loss = float('inf')
best_epoch = start_epoch
ompgs_cls_acc = 0
fsgs_cls_acc = 0
early_stop_counter = 0


# ===== 计算开销记录 =====
epoch_train_times = []
epoch_val_times = []
epoch_peak_memory = []
total_train_start = time.time()


for epoch in tqdm(range(start_epoch, n_epoch)):

    training_st = time.time()
    train_time = 0
    train_loss = 0

    adjust_learning_rate(lr, optimizer, epoch + 1)
    torch.cuda.reset_peak_memory_stats()  # 重置显存峰值统计
    _t0 = time.time()
    train_time, train_loss = train(model, train_loader, optimizer, epoch, log_train_test)
    epoch_train_times.append(time.time() - _t0)
    # epoch_peak_memory_train = torch.cuda.max_memory_allocated() / 1024 ** 2  # ← 紧跟train之后立即记录
    epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)  # ← 唯一的append，在train后、validate前
    if (epoch + 1) % 1 == 0:
        pre_at_sample = copy.deepcopy(next_at_sample)
        next_at_sample = [0 for i in range(len(samples_per_cls))]
        post_p = copy.deepcopy(piror_p)
        piror_p = [0 for i in range(len(samples_per_cls))]
    overall.append(copy.deepcopy(pre_at_sample))
    overall_error.append(copy.deepcopy(post_p))
    print("train pre_at_sample:", pre_at_sample)
    print("post_p:", post_p)
    print("train pre_at_sample:", pre_at_sample, file=log_train_test, flush=True)
    print("post_p:", post_p, file=log_train_test, flush=True)
    for i in range(class_num):
        writer.add_scalar('num/train pre_at_sample-{}'.format(i), pre_at_sample[i], epoch + 1)
        writer.add_scalar('num/post_p-{}'.format(i), post_p[i], epoch + 1)
    if 'wat' in args.alg: # our method

        _t1 = time.time()
        wat_valid_nat_cost, wat_valid_bndy_cost, val_rob_acc, val_rob_class_wise_acc, val_rob_auc = validate(model,
                                                                                                            valid_loader,
                                                                                                            wat_valid_nat_cost,
                                                                                                            wat_valid_bndy_cost)
        epoch_val_times.append(time.time() - _t1)
        # epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)  # 单位MB

        for i in range(class_num+1):
            wat_valid_cost[epoch, i] = wat_valid_nat_cost[epoch, i] + args.beta_wat * wat_valid_bndy_cost[epoch,i]  # 通过历史的nat和bndy cost来计算总的加权的cost
            if args.wat_wct:  # ← 加这个判断
                if args.wct_mode == 'cumsum':
                    # 原始做法：累积所有历史epoch的cost
                    class_factor = (torch.sum(wat_valid_cost, dim=0) * args.eta).exp()  # 对100个epoch的0类总权重加和，当分母
                # class_factor = (wat_valid_cost[epoch] * args.eta).exp()   # 这是新改的，只用当前epoch的cost，后来不用了
                elif args.wct_mode == 'current':
                    # 仅用当前epoch的cost
                    class_factor = (wat_valid_cost[epoch] * args.eta).exp()
                elif args.wct_mode == 'freq5':
                    # 每5个epoch才更新一次，其余epoch保持上一次的权重不变
                    if epoch % 5 == 0:
                        class_factor = (torch.sum(wat_valid_cost, dim=0) * args.eta).exp()
                    else:
                        class_factor = None  # 跳过更新
                if class_factor is not None:
                    nat_class_weights = class_num * class_factor / class_factor.sum()  # 求分子/分母更新 论文公式（12）
                    bndy_class_weights = class_num * class_factor / class_factor.sum()
            # wat_wct=False时，nat_class_weights/bndy_class_weights保持第135-136行初始化的全1值

            # print(f'Epoch: {epoch + 1} | Class-wise Cost: {wat_valid_cost[epoch, i].item():.8f}, Class-wise Weights: nat_class_weights[{i}]: {nat_class_weights.item():.8f}, bndy_class_weights[{i}]: {bndy_class_weights.item():.8f}')
            # print(f'Epoch: {epoch} | Class-wise Cost: {wat_valid_cost[epoch, i]:.8f}, Class-wise Weights: {nat_class_weights[i]:.8f}, {bndy_class_weights[i]:.8f}', file=log_train_test, flush=True)
            writer.add_scalar('loss/wat_valid_cost-{}'.format(i), wat_valid_cost[epoch, i], epoch + 1)
            writer.add_scalar('loss/wat_valid_nat_cost-{}'.format(i), wat_valid_nat_cost[epoch, i], epoch + 1)
            writer.add_scalar('loss/wat_valid_bndy_cost-{}'.format(i), wat_valid_bndy_cost[epoch, i], epoch + 1)
        # writer.add_scalar('loss/wat_valid_cost', wat_valid_cost[epoch, class_num], epoch + 1)
        # 在验证阶段打印每个类别的成本
        print(f"Epoch {epoch}: nat_class_weights (Class 0): {nat_class_weights[0]:.4f}, (Class 1): {nat_class_weights[1]:.4f}, (Sum 2): {nat_class_weights[2]:.4f}")
        print(f"Epoch {epoch}: nat_class_weights (Class 0): {nat_class_weights[0]:.4f}, (Class 1): {nat_class_weights[1]:.4f}, (Sum 2): {nat_class_weights[2]:.4f}", file=log_train_test, flush=True)
        print(f"Epoch {epoch}: bndy_class_weights (Class 0): {bndy_class_weights[0]:.4f}, (Class 1): {bndy_class_weights[1]:.4f}, (Sum 2): {bndy_class_weights[2]:.4f}")
        print(f"Epoch {epoch}: bndy_class_weights (Class 0): {bndy_class_weights[0]:.4f}, (Class 1): {bndy_class_weights[1]:.4f}, (Sum 2): {bndy_class_weights[2]:.4f}", file=log_train_test, flush=True)
        print(f"Epoch {epoch}: wat_valid_cost (Class 0): {wat_valid_cost[epoch, 0]:.4f}, (Class 1): {wat_valid_cost[epoch, 1]:.4f}, (Class 2): {wat_valid_cost[epoch, 2]:.4f}")
        print(f"Epoch {epoch}: wat_valid_cost (Class 0): {wat_valid_cost[epoch, 0]:.4f}, (Class 1): {wat_valid_cost[epoch, 1]:.4f}, (Class 2): {wat_valid_cost[epoch, 2]:.4f}", file=log_train_test, flush=True)
    elif 'none' in args.alg:  # vanilla TRADES

        _t1 = time.time()
        wat_valid_nat_cost, wat_valid_bndy_cost, val_rob_acc, val_rob_class_wise_acc, val_rob_auc = validate(model,
                                                                                                            valid_loader,
                                                                                                            wat_valid_nat_cost,
                                                                                                            wat_valid_bndy_cost)
        epoch_val_times.append(time.time() - _t1)
        # epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)  # 单位MB

        for i in range(class_num + 1):
            wat_valid_cost[epoch, i] = wat_valid_nat_cost[epoch, i] + args.beta_wat * wat_valid_bndy_cost[
                epoch, i]  # 通过历史的nat和bndy cost来计算总的加权的cost


            # print(f'Epoch: {epoch + 1} | Class-wise Cost: {wat_valid_cost[epoch, i].item():.8f}, Class-wise Weights: nat_class_weights[{i}]: {nat_class_weights.item():.8f}, bndy_class_weights[{i}]: {bndy_class_weights.item():.8f}')
            # print(f'Epoch: {epoch} | Class-wise Cost: {wat_valid_cost[epoch, i]:.8f}, Class-wise Weights: {nat_class_weights[i]:.8f}, {bndy_class_weights[i]:.8f}', file=log_train_test, flush=True)
            writer.add_scalar('loss/wat_valid_cost-{}'.format(i), wat_valid_cost[epoch, i], epoch + 1)
            writer.add_scalar('loss/wat_valid_nat_cost-{}'.format(i), wat_valid_nat_cost[epoch, i], epoch + 1)
            writer.add_scalar('loss/wat_valid_bndy_cost-{}'.format(i), wat_valid_bndy_cost[epoch, i], epoch + 1)
            writer.add_scalar('loss/val_total_cost_avg',
                              wat_valid_cost[epoch, class_num].item(), epoch + 1)
            writer.add_scalar('loss/val_nat_cost_avg',
                              wat_valid_nat_cost[epoch, class_num].item(), epoch + 1)
            writer.add_scalar('loss/val_bndy_cost_avg',
                              wat_valid_bndy_cost[epoch, class_num].item(), epoch + 1)
        # writer.add_scalar('loss/wat_valid_cost', wat_valid_cost[epoch, class_num], epoch + 1)
        # 在验证阶段打印每个类别的成本
        print(
            f"Epoch {epoch}: nat_class_weights (Class 0): {nat_class_weights[0]:.4f}, (Class 1): {nat_class_weights[1]:.4f}, (Sum 2): {nat_class_weights[2]:.4f}")
        print(
            f"Epoch {epoch}: nat_class_weights (Class 0): {nat_class_weights[0]:.4f}, (Class 1): {nat_class_weights[1]:.4f}, (Sum 2): {nat_class_weights[2]:.4f}",
            file=log_train_test, flush=True)
        print(
            f"Epoch {epoch}: bndy_class_weights (Class 0): {bndy_class_weights[0]:.4f}, (Class 1): {bndy_class_weights[1]:.4f}, (Sum 2): {bndy_class_weights[2]:.4f}")
        print(
            f"Epoch {epoch}: bndy_class_weights (Class 0): {bndy_class_weights[0]:.4f}, (Class 1): {bndy_class_weights[1]:.4f}, (Sum 2): {bndy_class_weights[2]:.4f}",
            file=log_train_test, flush=True)
        print(
            f"Epoch {epoch}: wat_valid_cost (Class 0): {wat_valid_cost[epoch, 0]:.4f}, (Class 1): {wat_valid_cost[epoch, 1]:.4f}, (Class 2): {wat_valid_cost[epoch, 2]:.4f}")
        print(
            f"Epoch {epoch}: wat_valid_cost (Class 0): {wat_valid_cost[epoch, 0]:.4f}, (Class 1): {wat_valid_cost[epoch, 1]:.4f}, (Class 2): {wat_valid_cost[epoch, 2]:.4f}",
            file=log_train_test, flush=True)

    elif args.alg == 'Normal':
        # val_loss, val_accuracy, val_class_wise_correct, val_auc_score = validate_Normal(model, valid_loader)
        _t1 = time.time()
        val_adv_loss, val_rob_acc, val_rob_class_wise_acc, val_rob_auc = validate_Normal(model, valid_loader)

        epoch_val_times.append(time.time() - _t1)
        # epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)  # 单位MB

    elif args.alg == 'PGD':
        if (val_clean_loss < best_val_nat_loss or val_adv_loss< best_val_bndy_loss)and epoch / n_epoch >=0.7:  # ) or epoch == 99
            best_val_loss = val_clean_loss
            best_val_bndy_loss = val_adv_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_nat_loss': best_val_loss,  # 记录当前验证损失
                'val_bndy_loss': best_val_bndy_loss,
                'test_loss': test_loss,
                'test_acc': test_nat_acc,
                'test_class_wise_acc': test_class_wise_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }, checkpoint=args.out_dir + '/ckpt/' + param_list, filename=str(epoch)+'.pt')
            print(
                f"Checkpoint saved at Epoch {epoch}, best_val_loss: {best_val_loss}")
        training_st = time.time() - training_st
        # use tensorboard to record infos
        for i in range(class_num):
            writer.add_scalar('test/test_class_wise_acc-{}'.format(i), test_class_wise_acc[i], epoch + 1)
            writer.add_scalar('val/val_rob_class_wise_acc-{}'.format(i), val_rob_class_wise_acc[i], epoch + 1)

        writer.add_scalar('worst/worst_test_acc', min(test_class_wise_acc), epoch + 1)
        writer.add_scalar('acc/test_nat_acc', test_nat_acc, epoch + 1)
        writer.add_scalar('acc/val_rob_acc', val_rob_acc, epoch + 1)
        writer.add_scalar('auc/val_auc', val_rob_auc, epoch + 1)
        writer.add_scalar('auc/test_auc', test_auc, epoch + 1)

        writer.add_scalar('loss/train_loss', train_loss / len(train_loader.dataset), epoch + 1)
        writer.add_scalar('loss/test_loss', test_loss, epoch + 1)
        writer.add_scalar('loss/val_adv_loss', val_adv_loss, epoch + 1)
        writer.add_scalar('loss/val_clean_loss', val_clean_loss, epoch + 1)

        writer.add_scalar('score/test_precision', test_precision, epoch + 1)
        writer.add_scalar('score/test_recall', test_recall, epoch + 1)
        writer.add_scalar('score/test_f1', test_f1, epoch + 1)

    else:   # REAT RBL BSL TAIL情况
        _t1 = time.time()
        val_adv_loss, val_clean_loss, val_rob_acc, val_rob_class_wise_acc, val_rob_auc = validate(model, valid_loader, wat_valid_nat_cost, wat_valid_bndy_cost)
        epoch_val_times.append(time.time() - _t1)
        # epoch_peak_memory.append(torch.cuda.max_memory_allocated() / 1024 ** 2)  # 单位MB

    ## Evalution
    torch.cuda.synchronize()  # 确保GPU计算完成再计时
    t_inf_start = time.time()
    test_loss, test_nat_acc, test_class_wise_acc, test_auc, test_precision, test_recall, test_f1 = attack.eval_clean(model, test_loader, Dataset, class_num=class_num)
    torch.cuda.synchronize()
    inference_time_per_batch = (time.time() - t_inf_start) / len(test_loader)  # 单位：秒/batch
    # print results
    print(f'Epoch: [{epoch + 1:d} | {n_epoch:d}] | Natural Test Loss {test_loss:.5f} | Natural Test Acc {test_nat_acc:.5f} '
          f'| Natural Test Class_Wise Acc {test_class_wise_acc} | Natural Test AUC {test_auc:.5f} | Natural Test Precision {test_precision:.5f} '
          f'| Natural Test Recall {test_recall:.5f} | Natural Test F1 {test_f1:.5f} |')  # test_pgd100_acc, test_cw_acc)
    print(
        f'Epoch: [{epoch + 1:d} | {n_epoch:d}] | Natural Test Loss {test_loss:.5f} | Natural Test Acc {test_nat_acc:.5f} '
        f'| Natural Test Class_Wise Acc {test_class_wise_acc} | Natural Test AUC {test_auc:.5f} | Natural Test Precision {test_precision:.5f} '
        f'| Natural Test Recall {test_recall:.5f} | Natural Test F1 {test_f1:.5f} |', file=log_train_test, flush=True)
    if args.alg == 'Normal':

        if val_adv_loss < best_val_nat_loss and epoch / n_epoch >=0.7:  # ) or epoch == 99
            best_val_loss = val_adv_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_nat_loss': best_val_loss,  # 记录当前验证损失
                'val_bndy_loss': 0,
                'test_loss': test_loss,
                'test_acc': test_nat_acc,
                'test_class_wise_acc': test_class_wise_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }, checkpoint=args.out_dir + '/ckpt/' + param_list, filename=str(epoch)+'.pt')
            print(
                f"Checkpoint saved at Epoch {epoch}, best_val_loss: {best_val_loss}")
        training_st = time.time() - training_st
        # use tensorboard to record infos
        for i in range(class_num):
            writer.add_scalar('test/test_class_wise_acc-{}'.format(i), test_class_wise_acc[i], epoch + 1)
            writer.add_scalar('val/val_class_wise_acc-{}'.format(i), val_rob_class_wise_acc[i], epoch + 1)

        writer.add_scalar('worst/worst_test_acc', min(test_class_wise_acc), epoch + 1)
        writer.add_scalar('acc/test_nat_acc', test_nat_acc, epoch + 1)
        writer.add_scalar('acc/val_rob_acc', val_rob_acc, epoch + 1)
        writer.add_scalar('auc/val_auc', val_rob_auc, epoch + 1)
        writer.add_scalar('auc/test_auc', test_auc, epoch + 1)

        writer.add_scalar('loss/train_loss', train_loss / len(train_loader.dataset), epoch + 1)
        writer.add_scalar('loss/test_loss', test_loss, epoch + 1)
        writer.add_scalar('score/test_precision', test_precision, epoch + 1)
        writer.add_scalar('score/test_recall', test_recall, epoch + 1)
        writer.add_scalar('score/test_f1', test_f1, epoch + 1)

    elif 'none' in args.alg or 'wat' in args.alg:
        import os
        log_attack_dir = os.path.join(args.out_dir, 'Logs', 'attack_utils')  # Construct the directory path
        os.makedirs(log_attack_dir, exist_ok=True)  # Create the directory, and any parent directories if needed

        log_attack_filename = f'Attack_{param_list}.bak'
        log_attack_filepath = os.path.join(log_attack_dir, log_attack_filename)

        log_attack = open(log_attack_filepath, 'a+')  # Now open the file, the directory exists
        log_attack = open(args.out_dir + f'/Logs/attack_utils/Attack_{param_list}.bak', 'a+')  # 'w+'是覆写模式 # ./resutls/{args.alg}/Logs/attack_utils
        # print(best_checkpoint_path, file=log_attack, flush=True)
        ompgs100_loss, test_ompgs100_acc, ompgs100_class_wise_acc, ompgs_acc, ompgs_auc, ompgs_precision, ompgs_recall, ompgs_f1 = attack.eval_robust(model,
                                                                                                            test_loader,
                                                                                                            perturb_steps=30,
                                                                                                            epsilon=epsilon,
                                                                                                            step_size=0.003,
                                                                                                            loss_fn="cw",
                                                                                                            category="OMPGS",
                                                                                                            rand_init=True,
                                                                                                            Dataset=Dataset,
                                                                                                            args=args,
                                                                                                            log_attack=log_attack,
                                                                                                            class_num=class_num)
        fsgs100_loss, test_fsgs100_acc, fsgs100_class_wise_acc, fsgs_acc, fsgs_auc, fsgs_precision, fsgs_recall, fsgs_f1 = attack.eval_robust(model,
                                                                                                       test_loader,
                                                                                                       perturb_steps=30,
                                                                                                       epsilon=epsilon,
                                                                                                       step_size=0.003,
                                                                                                       loss_fn="cw",
                                                                                                       category="FSGS",
                                                                                                       rand_init=True,
                                                                                                       Dataset=Dataset,
                                                                                                       args=args,
                                                                                                       log_attack=log_attack,
                                                                                                       class_num=class_num)
        # 假设每个 Epoch 结束后计算了 val_loss
        # 保存最佳epoch的模型
        if wat_valid_nat_cost[epoch, class_num] < best_val_nat_loss or wat_valid_nat_cost[epoch, class_num] < best_val_bndy_loss:
            early_stop_counter = 0

            if wat_valid_nat_cost[epoch, class_num] < best_val_nat_loss:
                best_val_nat_loss = wat_valid_nat_cost[epoch, class_num]  # 更新最小损失值

            if wat_valid_bndy_cost[epoch, class_num] < best_val_bndy_loss:  # 这里原来写的是wat_valid_nat_cost[epoch, class_num]
                best_val_bndy_loss = wat_valid_bndy_cost[epoch, class_num]

            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_nat_loss': best_val_nat_loss,  # 记录当前验证损失
                'val_bndy_loss': best_val_bndy_loss,
                'wat_valid_nat_cost': wat_valid_nat_cost,
                'wat_valid_bndy_cost': wat_valid_bndy_cost,
                'wat_valid_cost': wat_valid_cost,
                'test_loss': test_loss,
                'test_acc': test_nat_acc,
                'test_class_wise_acc': test_class_wise_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }, checkpoint=args.out_dir + '/ckpt/' + param_list, filename=str(epoch)+'.pt')  # 固定文件名，覆盖保存
            print(f"Checkpoint saved at Epoch {epoch}, best_val_nat_loss: {best_val_nat_loss}, best_val_bndy_loss: {best_val_bndy_loss}")
        # elif epoch < 30:
        #     early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 30:
                break

        training_st = time.time() - training_st
        # use tensorboard to record infos
        for i in range(class_num):
            writer.add_scalar('test/test_class_wise_acc-{}'.format(i), test_class_wise_acc[i], epoch + 1)
            writer.add_scalar('val/val_rob_class_wise_acc-{}'.format(i), val_rob_class_wise_acc[i], epoch + 1)
            writer.add_scalar('weight/nat-class-{}'.format(i), nat_class_weights[i].item(), epoch + 1)
            writer.add_scalar('weight/bndy-class-{}'.format(i), bndy_class_weights[i].item(), epoch + 1)

        writer.add_scalar('worst/worst_test_acc', min(test_class_wise_acc), epoch + 1)

        writer.add_scalar('acc/test_nat_acc', test_nat_acc, epoch + 1)
        writer.add_scalar('acc/val_rob_acc', val_rob_acc, epoch + 1)
        writer.add_scalar('auc/val_auc', val_rob_auc, epoch + 1)
        writer.add_scalar('auc/test_auc', test_auc, epoch + 1)

        writer.add_scalar('loss/train_loss', train_loss / len(train_loader.dataset), epoch + 1)
        writer.add_scalar('loss/test_loss', test_loss, epoch + 1)

        writer.add_scalar('loss/ompgs100_loss', ompgs100_loss, epoch + 1)
        writer.add_scalar('loss/fsgs100_loss', fsgs100_loss, epoch + 1)
        writer.add_scalar('auc/ompgs_auc', ompgs_auc, epoch + 1)
        writer.add_scalar('auc/fsgs_auc', fsgs_auc, epoch + 1)
        writer.add_scalar('acc/test_ompgs100_acc', test_ompgs100_acc, epoch + 1)
        writer.add_scalar('acc/test_fsgs100_acc', test_fsgs100_acc, epoch + 1)
        writer.add_scalar('worst/worst_ompgs100_acc', min(ompgs100_class_wise_acc), epoch + 1)
        writer.add_scalar('worst/worst_fsgs100_acc', min(fsgs100_class_wise_acc), epoch + 1)

        writer.add_scalar('score/ompgs_precision', ompgs_precision, epoch + 1)
        writer.add_scalar('score/ompgs_recall', ompgs_recall, epoch + 1)
        writer.add_scalar('score/ompgs_f1', ompgs_f1, epoch + 1)
        writer.add_scalar('score/fsgs_precision', fsgs_precision, epoch + 1)
        writer.add_scalar('score/fsgs_recall', fsgs_recall, epoch + 1)
        writer.add_scalar('score/fsgs_f1', fsgs_f1, epoch + 1)

    else:

        log_attack = open(args.out_dir + f'/Logs/attack_utils/Attack_{param_list}.bak',
                          'a+')  # 'w+'是覆写模式 # ./resutls/{args.alg}/Logs/attack_utils
        # print(best_checkpoint_path, file=log_attack, flush=True)
        ompgs100_loss, test_ompgs100_acc, ompgs100_class_wise_acc, ompgs_acc, ompgs_auc, ompgs_precision, ompgs_recall, ompgs_f1 = attack.eval_robust(model,
                                                                                                            test_loader,
                                                                                                            perturb_steps=30,
                                                                                                            epsilon=epsilon,
                                                                                                            step_size=0.003,
                                                                                                            loss_fn="cw",
                                                                                                            category="OMPGS",
                                                                                                            rand_init=True,
                                                                                                            Dataset=Dataset,
                                                                                                            args=args,
                                                                                                            log_attack=log_attack,
                                                                                                            class_num=class_num)
        fsgs100_loss, test_fsgs100_acc, fsgs100_class_wise_acc, fsgs_acc, fsgs_auc, fsgs_precision, fsgs_recall, fsgs_f1 = attack.eval_robust(model,
                                                                                                       test_loader,
                                                                                                       perturb_steps=30,
                                                                                                       epsilon=epsilon,
                                                                                                       step_size=0.003,
                                                                                                       loss_fn="cw",
                                                                                                       category="FSGS",
                                                                                                       rand_init=True,
                                                                                                       Dataset=Dataset,
                                                                                                       args=args,
                                                                                                       log_attack=log_attack,
                                                                                                       class_num=class_num)

        if ((min(ompgs100_class_wise_acc) > ompgs_cls_acc or min(fsgs100_class_wise_acc) > fsgs_cls_acc) and epoch>16) or epoch == 16: #   ) or epoch == 99 or epoch == 149
            early_stop_counter = 0
            ompgs_cls_acc = min(ompgs100_class_wise_acc)
            fsgs_cls_acc = min(fsgs100_class_wise_acc)
            best_val_loss = val_adv_loss
            best_clean_loss = val_clean_loss
            best_epoch = epoch
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_nat_loss': best_val_loss,  # 记录当前验证损失
                'val_bndy_loss': best_clean_loss,
                'test_loss': test_loss,
                'test_acc': test_nat_acc,
                'test_class_wise_acc': test_class_wise_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }, checkpoint=args.out_dir + '/ckpt/' + param_list, filename=str(epoch)+'.pt')
            print(
                f"Checkpoint saved at Epoch {epoch}, best_val_loss: {best_val_loss}")
        elif epoch < 16:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 30:
                break
        training_st = time.time() - training_st
        # use tensorboard to record infos
        for i in range(class_num):
            writer.add_scalar('test/test_class_wise_acc-{}'.format(i), test_class_wise_acc[i], epoch + 1)
            writer.add_scalar('val/val_rob_class_wise_acc-{}'.format(i), val_rob_class_wise_acc[i], epoch + 1)
            writer.add_scalar('ompgs/ompgs100_class_wise_acc-{}'.format(i), ompgs100_class_wise_acc[i], epoch + 1)
            writer.add_scalar('fsgs/fsgs100_class_wise_acc-{}'.format(i), fsgs100_class_wise_acc[i], epoch + 1)

        writer.add_scalar('worst/worst_test_acc', min(test_class_wise_acc), epoch + 1)

        writer.add_scalar('acc/test_nat_acc', test_nat_acc, epoch + 1)
        writer.add_scalar('acc/val_rob_acc', val_rob_acc, epoch + 1)

        writer.add_scalar('auc/val_auc', val_rob_auc, epoch + 1)
        writer.add_scalar('auc/test_auc', test_auc, epoch + 1)

        writer.add_scalar('loss/train_loss', train_loss / len(train_loader.dataset), epoch + 1)
        writer.add_scalar('loss/test_loss', test_loss, epoch + 1)

        writer.add_scalar('loss/val_adv_loss', val_adv_loss, epoch + 1)
        writer.add_scalar('loss/val_clean_loss', val_clean_loss, epoch + 1)


        writer.add_scalar('loss/ompgs100_loss', ompgs100_loss, epoch + 1)
        writer.add_scalar('loss/fsgs100_loss', fsgs100_loss, epoch + 1)
        writer.add_scalar('auc/ompgs_auc', ompgs_auc, epoch + 1)
        writer.add_scalar('auc/fsgs_auc', fsgs_auc, epoch + 1)
        writer.add_scalar('acc/test_ompgs100_acc', test_ompgs100_acc, epoch + 1)
        writer.add_scalar('acc/test_fsgs100_acc', test_fsgs100_acc, epoch + 1)
        writer.add_scalar('worst/worst_ompgs100_acc', min(ompgs100_class_wise_acc), epoch + 1)
        writer.add_scalar('worst/worst_fsgs100_acc', min(fsgs100_class_wise_acc), epoch + 1)

        writer.add_scalar('score/ompgs_precision', ompgs_precision, epoch + 1)
        writer.add_scalar('score/ompgs_recall', ompgs_recall, epoch + 1)
        writer.add_scalar('score/ompgs_f1', ompgs_f1, epoch + 1)
        writer.add_scalar('score/fsgs_precision', fsgs_precision, epoch + 1)
        writer.add_scalar('score/fsgs_recall', fsgs_recall, epoch + 1)
        writer.add_scalar('score/fsgs_f1', fsgs_f1, epoch + 1)


overall = np.array(overall).reshape((-1, num_classes[Dataset]))
np.save(args.out_dir+f'npyresult/{args.net}_{args.alg}_{batch_size}/{args.idx}_label_d.npy', overall)

overall_error = np.array(overall_error).reshape((-1, num_classes[Dataset]))
np.save(args.out_dir+f'npyresult/{args.net}_{args.alg}_{batch_size}/{args.idx}_error_d.npy', overall_error)


# 进行攻击测试
# 定义检查点路径
best_checkpoint_path = args.out_dir + '/ckpt/' + param_list + f'/{best_epoch}.pt'

print(best_checkpoint_path)
if not os.path.exists(args.out_dir + '/Logs/attack_utils/'):
    os.makedirs(args.out_dir + '/Logs/attack_utils/')
# log_attack = open(f'./Logs/{Dataset}/training/attack_utils/Attack_{Model_Name}_Adam_{lr}.bak', 'w+')
attack_param = param_list.split('/')[-1]
log_attack = open(args.out_dir + f'/Logs/attack_utils/Attack_{attack_param}.bak','a+')  # 'w+'是覆写模式 # ./resutls/{args.alg}/Logs/attack_utils
print(best_checkpoint_path, file=log_attack, flush=True)

# 加载检查点
checkpoint = torch.load(best_checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])

ompgs100_loss, test_ompgs100_acc, ompgs100_class_wise_acc, ompgs_acc, ompgs_auc, ompgs_precision, ompgs_recall, ompgs_f1 = attack.eval_robust(model, test_loader,perturb_steps=30, epsilon=epsilon, step_size=0.003,loss_fn="cw", category="OMPGS", rand_init=True, Dataset=Dataset, args=args, log_attack=log_attack, class_num=class_num)
fsgs100_loss, test_fsgs100_acc, fsgs100_class_wise_acc, fsgs_acc, fsgs_auc, fsgs_precision, fsgs_recall, fsgs_f1 = attack.eval_robust(model, test_loader,perturb_steps=30, epsilon=epsilon, step_size=0.003,loss_fn="cw", category="FSGS", rand_init=True, Dataset=Dataset, args=args, log_attack=log_attack, class_num=class_num)

# print results
print('OMPGS Test Acc %.5f Test AUC %.5f | FSGS Test Acc %.5f Test AUC %.5f |' % (test_ompgs100_acc, ompgs_auc, test_fsgs100_acc, fsgs_auc)) # test_pgd100_acc, test_cw_acc)
print('OMPGS Test precision %.5f Test recall %.5f Test f1 %.5f | FSGS Test precision %.5f Test recall %.5f Test f1 %.5f |' % (ompgs_precision, ompgs_recall, ompgs_f1, fsgs_precision, fsgs_recall, fsgs_f1))
print('OMPGS Test Acc %.5f Test AUC %.5f | FSGS Test Acc %.5f Test AUC %.5f |' % (test_ompgs100_acc, ompgs_auc, test_fsgs100_acc, fsgs_auc), file=log_attack, flush=True)
print('OMPGS Test precision %.5f Test recall %.5f Test f1 %.5f | FSGS Test precision %.5f Test recall %.5f Test f1 %.5f |' % (ompgs_precision, ompgs_recall, ompgs_f1, fsgs_precision, fsgs_recall, fsgs_f1), file=log_attack, flush=True)

# 获取训练参数
results = {
    "dataset": Dataset,
    "Dataset_type": Dataset_type[Dataset],
    "Model_Type": args.net,  # 模型类型
    "alg": args.alg,  # alg
    "wat_bsl": args.wat_bsl,  # 'False'
    "wat_tail": args.wat_tail, # 'False'
    "wat_rbl": args.wat_rbl,  # 'False'
    "wat_wct": args.wat_wct,
    "wct_mode": args.wct_mode if args.wat_wct else 'none',  # wat_mode
    "alg_new": '/',
    # 训练参数
    "idx": args.idx,
    "epoch": n_epoch,
    "lr": lr,
    "beta": args.beta if 'wat' in args.alg else '/', # beta
    "beta_wat": args.beta_wat if 'wat' in args.alg else '/',  # beta_wat
    "eta": args.eta if 'wat' in args.alg else '/',  # eta
    "delta_ratio": args.delta_ratio if 'wat' in args.alg else '/', # delta_ratio
    "num_steps": args.num_steps, #  num_steps
    "budgets/epsilon": epsilon,
    "scale": args.scale if 'wat' in args.alg else '/', # scale
    "alpha": '/',
    "theta": '/',
    # 训练结果
    "best_epoch": best_epoch,
    'val_nat_loss': checkpoint['val_nat_loss'],  # 记录当前验证损失 # , 'TAIL' .item() if all(x not in args.alg for x in ['Normal', 'REAT', 'RBL', 'BSL']) else checkpoint['val_nat_loss']
    'val_bndy_loss': checkpoint['val_bndy_loss'], # , 'TAIL'  .item() if all(x not in args.alg for x in ['Normal', 'REAT', 'RBL', 'BSL']) else checkpoint['val_bndy_loss']
    'test_nat_loss': checkpoint['test_loss'],
    'test_nat_acc': checkpoint['test_acc'],
    'test_nat_class_wise_acc': checkpoint['test_class_wise_acc'],
    'worst_nat_acc': min(checkpoint['test_class_wise_acc']),
    'test_nat_auc': checkpoint['test_auc'],
    'test_nat_precision': checkpoint['test_precision'],
    'test_nat_recall': checkpoint['test_recall'],
    'test_nat_f1': checkpoint['test_f1'],
    # 攻击后结果
    "acc_test_ompgs":test_ompgs100_acc,
    "acc_ompgs": ompgs_acc,
    "auc_ompgs": ompgs_auc,
    "ompgs_clsws_acc": ompgs100_class_wise_acc,
    "worst_ompgs_acc": min(ompgs100_class_wise_acc),
    "precision_ompgs": ompgs_precision,
    "recall_ompgs": ompgs_recall,
    "f1_ompgs": ompgs_f1,
    "acc_test_fsgs": test_fsgs100_acc,
    "acc_fsgs": fsgs_acc,
    "auc_fsgs": fsgs_auc,
    "fsgs_clsws_acc": fsgs100_class_wise_acc,
    "worst_fsgs_acc": min(fsgs100_class_wise_acc),
    "precision_fsgs": fsgs_precision,
    "recall_fsgs": fsgs_recall,
    "f1_fsgs": fsgs_f1,
    # 基础常数参数
    "test_size": test_sizes[Dataset],
    "num_feature": num_feature[Dataset],  # 输入特征数
    "num_classes": num_classes[Dataset],
    # 基础网络训练参数
    "batch_size": batch_size,  # 批次大小
    "emb_size": emb_sizes[Dataset],  # 嵌入层大小
    "num_new_feature": num_new_features[Dataset],  # 新特征数
    "hidden1": hidden1s[Dataset],  # 第一个隐藏层大小
    "hidden2": hidden2s[Dataset],  # 第二个隐藏层大小
    "hidden3": hidden3s[Dataset],  # 第三个隐藏层大小
    "batchnorm1d": batchnorm1ds[Dataset],  # 是否使用 BatchNorm1d
    "weight_decay": weight_decay,  # 权重衰减
    "beta_loss_new_opt": beta_loss_new_opts[Dataset],  # 新特征的损失权重

    # 攻击参数
    "OMPGS_time_limits": OMPGS_time_limits[Dataset],
    "FSGS_time_limits": FSGS_time_limits[Dataset],
    "PCAA_time_limits": PCAA_time_limits[Dataset],
    # 添加训练时间
    "train_time": training_st,  # 计算训练过程中的时间
    "param_time": param_time,  # 训练参数时间戳
    "best_checkpoint_path": best_checkpoint_path,
    "total_train_time_min": round((time.time() - total_train_start) / 60, 5),
    "avg_epoch_train_time_s": round(sum(epoch_train_times) / len(epoch_train_times), 5) if epoch_train_times else '/',
    "avg_epoch_val_time_s": round(sum(epoch_val_times) / len(epoch_val_times), 5) if epoch_val_times else '/',
    "peak_gpu_mem_MB": round(max(epoch_peak_memory), 5) if epoch_peak_memory else '/',
    "inference_time_per_batch": inference_time_per_batch

}

print(results)
output_log_dir = './results/'
# 将结果记录到文件
log_results_to_files(results, output_log_dir)

# 在训练结束后，额外再保存一个per-epoch明细csv（这个没法塞进results字典，因为是多行数据）：
import csv, os
cost_csv_path = os.path.join(args.out_dir, 'ckpt', param_list, 'epoch_cost_detail.csv')
with open(cost_csv_path, 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['epoch', 'train_time_s', 'val_time_s', 'peak_mem_MB'])
    for i, (tr, vl, mem) in enumerate(zip(epoch_train_times, epoch_val_times, epoch_peak_memory)):
        writer_csv.writerow([i, round(tr, 5), round(vl, 5), round(mem, 5)])
print(f"Epoch cost detail saved to {cost_csv_path}")