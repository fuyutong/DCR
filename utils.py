import numpy as np
import pandas as pd
import torch
import pickle
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from math import log
import copy
import torch.nn as nn
from advertorch.utils import replicate_input, is_float_or_torch_tensor, batch_multiply, batch_clamp, batch_l1_proj
from advertorch.attacks import Attack
from advertorch.attacks.utils import rand_init_delta, clamp, normalize_by_pnorm
import json
import csv

# Set random seed
seed = 42
np.random.seed(seed)  # NumPy random number generator seed
torch.manual_seed(seed)  # PyTorch random number generator seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # Set all GPU random seeds

# if test=True, load the test file, or load the whole file
def load_data(Dataset, test=True):
    if test:
        test_idx = pickle.load(open(Test_Idx_File[Dataset], 'rb'))
        whole_data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
        whole_label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
        data = whole_data[test_idx]
        label = whole_label[test_idx]
        return data, label
    data = pickle.load(open(Whole_Data_File[Dataset], 'rb'))
    label = pickle.load(open(Whole_Label_File[Dataset], 'rb'))
    return data, label


# load the dataset
def preparation(dataset):
    x = pickle.load(open('./dataset/' + dataset + 'X.pickle', 'rb'))
    y = pickle.load(open('./dataset/' + dataset + 'Y.pickle', 'rb'))
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y


test_sizes = {
    'Splice': 0.1,
    'pedec': 0.1,
    'census': 0.1,
    'stroke_multi': 0.4,   # Previously was 0.1
    'stroke_mixed': 0.4,
    'UC_multi': 0.4,
    'UC_mixed': 0.4,
    'UC_multi_13': 0.4,
    'UC_mixed_20': 0.4,
    'Thyroid_multi': 0.4,  # Originally intended 0.3, but code actually splits to 0.4
    'Thyroid_mixed': 0.4,
    'cardio_multi': 0.3,
    'cardio_mixed': 0.3,
    'wids_mixed': 0.3,
    'diatri_mixed': 0.3,
    'diatri_multi': 0.3,
    'UC_multi_balanced':0.4,
    'UC_multi_imbalanced':0.4,
    'Thyroid_multi_balanced': 0.4
}


def dataset_split(Dataset):
    X, y = preparation(Dataset)
    X_np = X.numpy()
    y_np = y.numpy()
    df_x = pd.DataFrame(X_np)
    df_y = pd.Series(y_np)
    X_train, X_test, y_train, y_test = train_test_split(df_x,df_y, test_size=test_sizes[Dataset], stratify=y, random_state=seed)
    train_idx = X_train.index
    test_idx = X_test.index

    return train_idx, test_idx


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def tune_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


def input_process(input_data, Dataset):
    """
    Process input data for different dataset types
    """
    if Dataset_type[Dataset] == 'mixed':
        # Mixed type data: [continuous features, categorical features]
        continuous_part = input_data[:, :num_con_feature[Dataset]].float()
        categorical_part = input_data[:, num_con_feature[Dataset]:].long()

        # Apply one-hot encoding to categorical part
        categorical_encoded = []
        for i in range(categorical_part.shape[1]):
            one_hot = torch.zeros(categorical_part.shape[0], num_category[Dataset])
            one_hot.scatter_(1, categorical_part[:, i:i+1], 1)
            categorical_encoded.append(one_hot)

        categorical_encoded = torch.stack(categorical_encoded, dim=1)
        return [continuous_part.cuda(), categorical_encoded.cuda()]
    elif Dataset_type[Dataset] == 'multi':
        # Multi-category data: one-hot encode each feature
        input_long = input_data.long()
        encoded_data = torch.zeros(input_data.shape[0], input_data.shape[1], num_category[Dataset])
        for i in range(input_data.shape[1]):
            encoded_data[:, i, :] = torch.eye(num_category[Dataset])[input_long[:, i]]
        return encoded_data.cuda()
    elif Dataset_type[Dataset] == 'binary':
        # Binary data: keep as is but convert to float
        return input_data.float().cuda()
    else:
        # Default case
        return input_data.float().cuda()


def one_hot_samples(input_data, Dataset):
    """
    One-hot encode input samples for multi-category data
    """
    if Dataset_type[Dataset] == 'multi':
        input_long = input_data.long()
        encoded_data = torch.zeros(input_data.shape[0], input_data.shape[1], num_category[Dataset])
        for i in range(input_data.shape[1]):
            encoded_data[:, i, :] = torch.eye(num_category[Dataset])[input_long[:, i]]
        return encoded_data
    else:
        return input_data


# Dataset configurations
Dataset_type = {
    'Splice': 'multi',
    'pedec': 'multi',
    'census': 'multi',
    'stroke_multi': 'multi',
    'stroke_mixed': 'mixed',
    'UC_multi': 'multi',
    'UC_mixed': 'mixed',
    'UC_multi_13': 'multi',
    'UC_mixed_20': 'mixed',
    'Thyroid_multi': 'multi',
    'Thyroid_mixed': 'mixed',
    'cardio_multi': 'multi',
    'cardio_mixed': 'mixed',
    'wids_mixed': 'mixed',
    'diatri_mixed': 'mixed',
    'diatri_multi': 'multi',
    'UC_multi_balanced': 'multi',
    'UC_multi_imbalanced': 'multi',
    'Thyroid_multi_balanced': 'multi'
}

num_classes = {
    'Splice': 3,
    'pedec': 2,
    'census': 2,
    'stroke_multi': 2,
    'stroke_mixed': 2,
    'UC_multi': 2,
    'UC_mixed': 2,
    'UC_multi_13': 13,
    'UC_mixed_20': 20,
    'Thyroid_multi': 4,
    'Thyroid_mixed': 4,
    'cardio_multi': 2,
    'cardio_mixed': 2,
    'wids_mixed': 2,
    'diatri_mixed': 2,
    'diatri_multi': 2,
    'UC_multi_balanced': 2,
    'UC_multi_imbalanced': 2,
    'Thyroid_multi_balanced': 4
}

num_feature = {
    'Splice': 60,
    'pedec': 5000,
    'census': 32,
    'stroke_multi': 10,
    'stroke_mixed': 15,
    'UC_multi': 39,
    'UC_mixed': 44,
    'UC_multi_13': 39,
    'UC_mixed_20': 44,
    'Thyroid_multi': 17,
    'Thyroid_mixed': 22,
    'cardio_multi': 23,
    'cardio_mixed': 28,
    'wids_mixed': 250,
    'diatri_mixed': 4,
    'diatri_multi': 4,
    'UC_multi_balanced': 39,
    'UC_multi_imbalanced': 39,
    'Thyroid_multi_balanced': 17
}

num_category = {
    'Splice': 5,
    'pedec': 3,
    'census': 52,
    'stroke_multi': 3,
    'stroke_mixed': 3,
    'UC_multi': 10,
    'UC_mixed': 10,
    'UC_multi_13': 10,
    'UC_mixed_20': 10,
    'Thyroid_multi': 2,
    'Thyroid_mixed': 2,
    'cardio_multi': 16,
    'cardio_mixed': 16,
    'wids_mixed': 10,
    'diatri_mixed': 10,
    'diatri_multi': 10,
    'UC_multi_balanced': 10,
    'UC_multi_imbalanced': 10,
    'Thyroid_multi_balanced': 2
}

emb_sizes = {
    'Splice': 30,
    'pedec': 1500,
    'census': 16,
    'stroke_multi': 15,
    'stroke_mixed': 15,
    'UC_multi': 20,
    'UC_mixed': 20,
    'UC_multi_13': 20,
    'UC_mixed_20': 20,
    'Thyroid_multi': 10,
    'Thyroid_mixed': 10,
    'cardio_multi': 20,
    'cardio_mixed': 20,
    'wids_mixed': 50,
    'diatri_mixed': 5,
    'diatri_multi': 5,
    'UC_multi_balanced': 20,
    'UC_multi_imbalanced': 20,
    'Thyroid_multi_balanced': 10
}

hidden1s = {
    'Splice': 128,
    'pedec': 256,
    'census': 128,
    'stroke_multi': 128,
    'stroke_mixed': 128,
    'UC_multi': 128,
    'UC_mixed': 128,
    'UC_multi_13': 128,
    'UC_mixed_20': 128,
    'Thyroid_multi': 128,
    'Thyroid_mixed': 128,
    'cardio_multi': 128,
    'cardio_mixed': 128,
    'wids_mixed': 256,
    'diatri_mixed': 64,
    'diatri_multi': 64,
    'UC_multi_balanced': 128,
    'UC_multi_imbalanced': 128,
    'Thyroid_multi_balanced': 128
}

hidden2s = {
    'Splice': 128,
    'pedec': 128,
    'census': 64,
    'stroke_multi': 64,
    'stroke_mixed': 64,
    'UC_multi': 64,
    'UC_mixed': 64,
    'UC_multi_13': 64,
    'UC_mixed_20': 64,
    'Thyroid_multi': 64,
    'Thyroid_mixed': 64,
    'cardio_multi': 64,
    'cardio_mixed': 64,
    'wids_mixed': 128,
    'diatri_mixed': 32,
    'diatri_multi': 32,
    'UC_multi_balanced': 64,
    'UC_multi_imbalanced': 64,
    'Thyroid_multi_balanced': 64
}

hidden3s = {
    'Splice': 0,
    'pedec': 0,
    'census': 0,
    'stroke_multi': 0,
    'stroke_mixed': 0,
    'UC_multi': 0,
    'UC_mixed': 0,
    'UC_multi_13': 0,
    'UC_mixed_20': 0,
    'Thyroid_multi': 0,
    'Thyroid_mixed': 0,
    'cardio_multi': 0,
    'cardio_mixed': 0,
    'wids_mixed': 0,
    'diatri_mixed': 0,
    'diatri_multi': 0,
    'UC_multi_balanced': 0,
    'UC_multi_imbalanced': 0,
    'Thyroid_multi_balanced': 0
}

batchnorm1ds = {
    'Splice': True,
    'pedec': True,
    'census': True,
    'stroke_multi': True,
    'stroke_mixed': True,
    'UC_multi': True,
    'UC_mixed': True,
    'UC_multi_13': True,
    'UC_mixed_20': True,
    'Thyroid_multi': True,
    'Thyroid_mixed': True,
    'cardio_multi': True,
    'cardio_mixed': True,
    'wids_mixed': True,
    'diatri_mixed': True,
    'diatri_multi': True,
    'UC_multi_balanced': True,
    'UC_multi_imbalanced': True,
    'Thyroid_multi_balanced': True
}

num_new_features = {
    'Splice': 20,
    'pedec': 20,
    'census': 16,
    'stroke_multi': 10,
    'stroke_mixed': 10,
    'UC_multi': 20,
    'UC_mixed': 20,
    'UC_multi_13': 20,
    'UC_mixed_20': 20,
    'Thyroid_multi': 10,
    'Thyroid_mixed': 10,
    'cardio_multi': 15,
    'cardio_mixed': 15,
    'wids_mixed': 25,
    'diatri_mixed': 4,
    'diatri_multi': 4,
    'UC_multi_balanced': 20,
    'UC_multi_imbalanced': 20,
    'Thyroid_multi_balanced': 10
}

# Continuous feature counts for mixed datasets
num_con_feature = {
    'stroke_mixed': 5,
    'UC_mixed': 5,
    'UC_mixed_20': 5,
    'Thyroid_mixed': 7,
    'cardio_mixed': 8,
    'wids_mixed': 50,
    'diatri_mixed': 2
}

# Define complex categories (mixed datasets with different category counts per feature)
complex_categories = {
    'stroke_mixed': [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'UC_mixed': [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'UC_mixed_20': [3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'Thyroid_mixed': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'cardio_mixed': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    'wids_mixed': [10] * 50 + [2] * 200,
    'diatri_mixed': [3, 3, 2, 2]
}

num_avail_category = {
    'Splice': 5,
    'pedec': 3,
    'census': 52,
    'stroke_multi': 3,
    'stroke_mixed': 3,
    'UC_multi': 10,
    'UC_mixed': 10,
    'UC_multi_13': 10,
    'UC_mixed_20': 10,
    'Thyroid_multi': 2,
    'Thyroid_mixed': 2,
    'cardio_multi': 16,
    'cardio_mixed': 16,
    'wids_mixed': 10,
    'diatri_mixed': 10,
    'diatri_multi': 10,
    'UC_multi_balanced': 10,
    'UC_multi_imbalanced': 10,
    'Thyroid_multi_balanced': 2
}

# Training parameters for different datasets
lr_list = {
    'Splice': 0.001,
    'pedec': 0.001,
    'census': 0.001,
    'stroke_multi': 0.001,
    'stroke_mixed': 0.001,
    'UC_multi': 0.001,
    'UC_mixed': 0.001,
    'UC_multi_13': 0.001,
    'UC_mixed_20': 0.001,
    'Thyroid_multi': 0.001,
    'Thyroid_mixed': 0.001,
    'cardio_multi': 0.001,
    'cardio_mixed': 0.001,
    'wids_mixed': 0.0001,
    'diatri_mixed': 0.001,
    'diatri_multi': 0.001,
    'UC_multi_balanced': 0.001,
    'UC_multi_imbalanced': 0.001,
    'Thyroid_multi_balanced': 0.001
}

epochs = {
    'Splice': 100,
    'pedec': 100,
    'census': 100,
    'stroke_multi': 100,
    'stroke_mixed': 100,
    'UC_multi': 100,
    'UC_mixed': 100,
    'UC_multi_13': 100,
    'UC_mixed_20': 100,
    'Thyroid_multi': 100,
    'Thyroid_mixed': 100,
    'cardio_multi': 100,
    'cardio_mixed': 100,
    'wids_mixed': 50,
    'diatri_mixed': 100,
    'diatri_multi': 100,
    'UC_multi_balanced': 100,
    'UC_multi_imbalanced': 100,
    'Thyroid_multi_balanced': 100
}

batch_sizes = {
    'Splice': 32,
    'pedec': 32,
    'census': 32,
    'stroke_multi': 32,
    'stroke_mixed': 32,
    'UC_multi': 32,
    'UC_mixed': 32,
    'UC_multi_13': 32,
    'UC_mixed_20': 32,
    'Thyroid_multi': 32,
    'Thyroid_mixed': 32,
    'cardio_multi': 32,
    'cardio_mixed': 32,
    'wids_mixed': 16,
    'diatri_mixed': 32,
    'diatri_multi': 32,
    'UC_multi_balanced': 32,
    'UC_multi_imbalanced': 32,
    'Thyroid_multi_balanced': 32
}

budgets = {
    'Splice': 0.031,
    'pedec': 0.031,
    'census': 0.031,
    'stroke_multi': 0.031,
    'stroke_mixed': 0.031,
    'UC_multi': 0.031,
    'UC_mixed': 0.031,
    'UC_multi_13': 0.031,
    'UC_mixed_20': 0.031,
    'Thyroid_multi': 0.031,
    'Thyroid_mixed': 0.031,
    'cardio_multi': 0.031,
    'cardio_mixed': 0.031,
    'wids_mixed': 0.031,
    'diatri_mixed': 0.031,
    'diatri_multi': 0.031,
    'UC_multi_balanced': 0.031,
    'UC_multi_imbalanced': 0.031,
    'Thyroid_multi_balanced': 0.031
}

weight_decays = {
    'Splice': 5e-4,
    'pedec': 5e-4,
    'census': 5e-4,
    'stroke_multi': 5e-4,
    'stroke_mixed': 5e-4,
    'UC_multi': 5e-4,
    'UC_mixed': 5e-4,
    'UC_multi_13': 5e-4,
    'UC_mixed_20': 5e-4,
    'Thyroid_multi': 5e-4,
    'Thyroid_mixed': 5e-4,
    'cardio_multi': 5e-4,
    'cardio_mixed': 5e-4,
    'wids_mixed': 5e-4,
    'diatri_mixed': 5e-4,
    'diatri_multi': 5e-4,
    'UC_multi_balanced': 5e-4,
    'UC_multi_imbalanced': 5e-4,
    'Thyroid_multi_balanced': 5e-4
}

beta_loss_new_opts = {
    'Splice': True,
    'pedec': True,
    'census': True,
    'stroke_multi': True,
    'stroke_mixed': True,
    'UC_multi': True,
    'UC_mixed': True,
    'UC_multi_13': True,
    'UC_mixed_20': True,
    'Thyroid_multi': True,
    'Thyroid_mixed': True,
    'cardio_multi': True,
    'cardio_mixed': True,
    'wids_mixed': True,
    'diatri_mixed': True,
    'diatri_multi': True,
    'UC_multi_balanced': True,
    'UC_multi_imbalanced': True,
    'Thyroid_multi_balanced': True
}

# Time limits for attacks
OMPGS_time_limits = {
    'Splice': 100,
    'pedec': 100,
    'census': 100,
    'stroke_multi': 100,
    'stroke_mixed': 100,
    'UC_multi': 100,
    'UC_mixed': 100,
    'UC_multi_13': 100,
    'UC_mixed_20': 100,
    'Thyroid_multi': 100,
    'Thyroid_mixed': 100,
    'cardio_multi': 100,
    'cardio_mixed': 100,
    'wids_mixed': 100,
    'diatri_mixed': 100,
    'diatri_multi': 100,
    'UC_multi_balanced': 100,
    'UC_multi_imbalanced': 100,
    'Thyroid_multi_balanced': 100
}

FSGS_time_limits = {
    'Splice': 100,
    'pedec': 100,
    'census': 100,
    'stroke_multi': 100,
    'stroke_mixed': 100,
    'UC_multi': 100,
    'UC_mixed': 100,
    'UC_multi_13': 100,
    'UC_mixed_20': 100,
    'Thyroid_multi': 100,
    'Thyroid_mixed': 100,
    'cardio_multi': 100,
    'cardio_mixed': 100,
    'wids_mixed': 100,
    'diatri_mixed': 100,
    'diatri_multi': 100,
    'UC_multi_balanced': 100,
    'UC_multi_imbalanced': 100,
    'Thyroid_multi_balanced': 100
}

PCAA_time_limits = {
    'Splice': 100,
    'pedec': 100,
    'census': 100,
    'stroke_multi': 100,
    'stroke_mixed': 100,
    'UC_multi': 100,
    'UC_mixed': 100,
    'UC_multi_13': 100,
    'UC_mixed_20': 100,
    'Thyroid_multi': 100,
    'Thyroid_mixed': 100,
    'cardio_multi': 100,
    'cardio_mixed': 100,
    'wids_mixed': 100,
    'diatri_mixed': 100,
    'diatri_multi': 100,
    'UC_multi_balanced': 100,
    'UC_multi_imbalanced': 100,
    'Thyroid_multi_balanced': 100
}

# TabNet parameters for different datasets
tabnet_n_steps = {
    'Splice': 3,
    'pedec': 3,
    'census': 3,
    'stroke_multi': 3,
    'stroke_mixed': 3,
    'UC_multi': 3,
    'UC_mixed': 3,
    'UC_multi_13': 3,
    'UC_mixed_20': 3,
    'Thyroid_multi': 3,
    'Thyroid_mixed': 3,
    'cardio_multi': 3,
    'cardio_mixed': 3,
    'wids_mixed': 3,
    'diatri_mixed': 3,
    'diatri_multi': 3,
    'UC_multi_balanced': 3,
    'UC_multi_imbalanced': 3,
    'Thyroid_multi_balanced': 3
}

tabnet_n_d = {
    'Splice': 32,
    'pedec': 32,
    'census': 32,
    'stroke_multi': 32,
    'stroke_mixed': 32,
    'UC_multi': 32,
    'UC_mixed': 32,
    'UC_multi_13': 32,
    'UC_mixed_20': 32,
    'Thyroid_multi': 32,
    'Thyroid_mixed': 32,
    'cardio_multi': 32,
    'cardio_mixed': 32,
    'wids_mixed': 32,
    'diatri_mixed': 32,
    'diatri_multi': 32,
    'UC_multi_balanced': 32,
    'UC_multi_imbalanced': 32,
    'Thyroid_multi_balanced': 32
}

tabnet_n_a = {
    'Splice': 32,
    'pedec': 32,
    'census': 32,
    'stroke_multi': 32,
    'stroke_mixed': 32,
    'UC_multi': 32,
    'UC_mixed': 32,
    'UC_multi_13': 32,
    'UC_mixed_20': 32,
    'Thyroid_multi': 32,
    'Thyroid_mixed': 32,
    'cardio_multi': 32,
    'cardio_mixed': 32,
    'wids_mixed': 32,
    'diatri_mixed': 32,
    'diatri_multi': 32,
    'UC_multi_balanced': 32,
    'UC_multi_imbalanced': 32,
    'Thyroid_multi_balanced': 32
}

tabnet_gamma = {
    'Splice': 1.3,
    'pedec': 1.3,
    'census': 1.3,
    'stroke_multi': 1.3,
    'stroke_mixed': 1.3,
    'UC_multi': 1.3,
    'UC_mixed': 1.3,
    'UC_multi_13': 1.3,
    'UC_mixed_20': 1.3,
    'Thyroid_multi': 1.3,
    'Thyroid_mixed': 1.3,
    'cardio_multi': 1.3,
    'cardio_mixed': 1.3,
    'wids_mixed': 1.3,
    'diatri_mixed': 1.3,
    'diatri_multi': 1.3,
    'UC_multi_balanced': 1.3,
    'UC_multi_imbalanced': 1.3,
    'Thyroid_multi_balanced': 1.3
}


def log_results_to_files(results, output_log_dir):
    """Log results to CSV file."""
    os.makedirs(output_log_dir, exist_ok=True)

    # Prepare the output CSV file path
    csv_file_path = os.path.join(output_log_dir, f"{results['dataset']}_{results['alg']}_results.csv")

    # Extract the keys from the results dictionary
    keys = results.keys()

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        # If the CSV file doesn't exist, create it with headers
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=keys)
            writer.writeheader()

    # Append the results to the CSV file
    with open(csv_file_path, 'a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writerow(results)


# Path definitions (these would normally point to actual data files)
Test_Idx_File = {
    'Splice': './dataset/Splice_test_idx.pickle',
    'pedec': './dataset/pedec_test_idx.pickle',
    'census': './dataset/census_test_idx.pickle',
    'stroke_multi': './dataset/stroke_multi_test_idx.pickle',
    'stroke_mixed': './dataset/stroke_mixed_test_idx.pickle',
    'UC_multi': './dataset/UC_multi_test_idx.pickle',
    'UC_mixed': './dataset/UC_mixed_test_idx.pickle',
    'UC_multi_13': './dataset/UC_multi_13_test_idx.pickle',
    'UC_mixed_20': './dataset/UC_mixed_20_test_idx.pickle',
    'Thyroid_multi': './dataset/Thyroid_multi_test_idx.pickle',
    'Thyroid_mixed': './dataset/Thyroid_mixed_test_idx.pickle',
    'cardio_multi': './dataset/cardio_multi_test_idx.pickle',
    'cardio_mixed': './dataset/cardio_mixed_test_idx.pickle'
}


Train_Idx_File = {
    'Splice': './dataset/Splice_train_idx.pickle',
    'pedec': './dataset/pedec_train_idx.pickle',
    'census': './dataset/census_train_idx.pickle',
    'stroke_multi': './dataset/stroke_multi_train_idx.pickle',
    'stroke_mixed': './dataset/stroke_mixed_train_idx.pickle',
    'UC_multi': './dataset/UC_multi_train_idx.pickle',
    'UC_mixed': './dataset/UC_mixed_train_idx.pickle',
    'UC_multi_13': './dataset/UC_multi_13_train_idx.pickle',
    'UC_mixed_20': './dataset/UC_mixed_20_train_idx.pickle',
    'Thyroid_multi': './dataset/Thyroid_multi_train_idx.pickle',
    'Thyroid_mixed': './dataset/Thyroid_mixed_train_idx.pickle',
    'cardio_multi': './dataset/cardio_multi_train_idx.pickle',
    'cardio_mixed': './dataset/cardio_mixed_train_idx.pickle'
}


Whole_Data_File = {
    'Splice': './dataset/SpliceX.pickle',
    'pedec': './dataset/pedecX.pickle',
    'census': './dataset/censusX.pickle',
    'stroke_multi': './dataset/stroke_multiX.pickle',
    'stroke_mixed': './dataset/stroke_mixedX.pickle',
    'UC_multi': './dataset/UC_multiX.pickle',
    'UC_mixed': './dataset/UC_mixedX.pickle',
    'UC_multi_13': './dataset/UC_multi_13X.pickle',
    'UC_mixed_20': './dataset/UC_mixed_20X.pickle',
    'Thyroid_multi': './dataset/Thyroid_multiX.pickle',
    'Thyroid_mixed': './dataset/Thyroid_mixedX.pickle',
    'cardio_multi': './dataset/cardio_multiX.pickle',
    'cardio_mixed': './dataset/cardio_mixedX.pickle',
    'UC_multi_balanced': './dataset/UC_multi_balancedX.pickle',
    'UC_multi_imbalanced': './dataset/UC_multi_imbalancedX.pickle'
}


Whole_Label_File = {
    'Splice': './dataset/SpliceY.pickle',
    'pedec': './dataset/pedecY.pickle',
    'census': './dataset/censusY.pickle',
    'stroke_multi': './dataset/stroke_multiY.pickle',
    'stroke_mixed': './dataset/stroke_mixedY.pickle',
    'UC_multi': './dataset/UC_multiY.pickle',
    'UC_mixed': './dataset/UC_mixedY.pickle',
    'UC_multi_13': './dataset/UC_multi_13Y.pickle',
    'UC_mixed_20': './dataset/UC_mixed_20Y.pickle',
    'Thyroid_multi': './dataset/Thyroid_multiY.pickle',
    'Thyroid_mixed': './dataset/Thyroid_mixedY.pickle',
    'cardio_multi': './dataset/cardio_multiY.pickle',
    'cardio_mixed': './dataset/cardio_mixedY.pickle',
    'UC_multi_balanced': './dataset/UC_multi_balancedY.pickle',
    'UC_multi_imbalanced': './dataset/UC_multi_imbalancedY.pickle'
}