# Data loader for WAT (Worst-class Adversarial Training)
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import torch

# Set random seed
seed = 42
np.random.seed(seed)  # NumPy random number generator seed
torch.manual_seed(seed)  # PyTorch random number generator seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)  # Set all GPU random seeds


class Cifar10():
    def __init__(self, mode="train"):

        self.mode = mode

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.transform_valid = transforms.Compose([
            transforms.ToTensor(),
        ])


        if self.mode == 'train' or self.mode == 'valid':

            self.cifar10 = datasets.CIFAR10('./data', train=True, download=True)

            data_source = self.cifar10.data
            label_source = self.cifar10.targets
            label_source = np.array(label_source)

            self.data = []
            self.labels = []
            classes = range(10)

            ## training data
            if self.mode == 'train':
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[0:4700]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[0:4700]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

            elif self.mode == 'valid': ## validation data

                classes = range(10)
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[4700:5000]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[4700:5000]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

        elif self.mode == 'test':
            self.cifar10 = datasets.CIFAR10('./data', train=False, download=True)
            self.data = self.cifar10.data
            self.labels = self.cifar10.targets


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        img = self.data[index]
        target =  self.labels[index]


        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.mode == 'train':
            img = self.transform(img)
        elif self.mode == 'valid':
            img = self.transform_valid(img)
        elif self.mode == 'test':
            img = self.transform_valid(img)
        return img, target


class Cifar100():
    def __init__(self, mode="train"):

        self.mode = mode

        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])

        self.transform_valid = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
        ])


        if self.mode == 'train' or self.mode == 'valid':

            self.cifar100 = datasets.CIFAR100('./data', train=True, download=True)

            data_source = self.cifar100.data
            label_source = self.cifar100.targets
            label_source = np.array(label_source)

            self.data = []
            self.labels = []
            classes = range(100)

            ## training data
            if self.mode == 'train':
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[0:470]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[0:470]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

            elif self.mode == 'valid': ## validation data

                classes = range(100)
                for c in classes:
                    tmp_idx = np.where(label_source == c)[0]
                    img = data_source[tmp_idx[470:500]]
                    self.data.append(img)
                    cl = label_source[tmp_idx[470:500]]
                    self.labels.append(cl)

                self.data = np.concatenate(self.data)
                self.labels = np.concatenate(self.labels)

        elif self.mode == 'test':
            self.cifar100 = datasets.CIFAR100('./data', train=False, download=True)
            self.data = self.cifar100.data
            self.labels = self.cifar100.targets


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        img = self.data[index]
        target =  self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        img = Image.fromarray(img)

        if self.mode == 'train':
            img = self.transform(img)
        elif self.mode == 'valid':
            img = self.transform_valid(img)
        elif self.mode == 'test':
            img = self.transform_valid(img)
        return img, target


def get_cifar10_loader(batch_size):


    """Build and return data loader."""

    dataset1 = Cifar10(mode='train')
    dataset2 = Cifar10(mode='valid')
    dataset3 = Cifar10(mode='test')

    train_loader = DataLoader(dataset=dataset1,
                             batch_size=batch_size,
                             shuffle=True)

    valid_loader = DataLoader(dataset=dataset2,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset3,
                              batch_size=batch_size,
                              shuffle=True)
    return train_loader, valid_loader, test_loader


def get_cifar100_loader(batch_size):


    """Build and return data loader."""

    dataset1 = Cifar100(mode='train')
    dataset2 = Cifar100(mode='valid')
    dataset3 = Cifar100(mode='test')

    train_loader = DataLoader(dataset=dataset1,
                             batch_size=batch_size,
                             shuffle=True)

    valid_loader = DataLoader(dataset=dataset2,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset3,
                              batch_size=batch_size,
                              shuffle=True)
    return train_loader, valid_loader, test_loader


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


class TabularDataset():
    def __init__(self, mode="train", dataset_name=None):
        self.mode = mode
        self.dataset_name = dataset_name

        # Load full data
        X, y = preparation(dataset_name)  # X: [num_samples, 39], y: [num_samples]

        # Get indices for training, validation, and test sets
        train_idx, test_idx = dataset_split(dataset_name)

        # Split training and validation sets (divide training data 8:2)
        if self.mode == 'train':
            self.data = X[train_idx]
            self.labels = y[train_idx]

            unique_labels, counts = np.unique(self.labels, return_counts=True)

            # Create a dictionary mapping class labels to counts
            class_counts_dict = dict(zip(unique_labels, counts))
            self.class_counts_dict = class_counts_dict
            print("self.class_counts_dict: ",self.class_counts_dict)
            # Assume your dataset has num_classes classes, with class indices 0, 1, 2, ...
            num_classes = len(unique_labels)  # Or get num_classes from your dataset definition
            self.samples_per_cls = []
            for class_index in range(num_classes):
                # Look up the count corresponding to class index in the dictionary, if not in dictionary then count is 0 (meaning no samples in training set for this class)
                count = class_counts_dict.get(class_index, 0)  # .get(key, default_value) method
                self.samples_per_cls.append(count)

            # Now self.samples_per_cls list index i corresponds to sample count for class index i
        elif self.mode == 'valid':
            X_train, X_valid, y_train, y_valid = train_test_split(
                X[train_idx], y[train_idx],
                test_size=0.1,  # 10% as validation set
                stratify=y[train_idx],
                random_state=seed
            )
            self.data = X_valid
            self.labels = y_valid
        else:  # test
            self.data = X[test_idx]
            self.labels = y[test_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Directly return tensors, no image transformations needed
        features = self.data[index].float()  # Convert to float32
        label = self.labels[index].long()  # Convert to int64
        return features, label

    def get_samples_per_cls(self): # Add method to get samples_per_cls
        if self.mode == 'train':
            return self.samples_per_cls
        else:
            return None # Or return an empty list, or raise exception depending on your needs


def get_multi_loader(batch_size, Dataset):
    """Build and return data loader for multi-dataset"""
    # Initialize dataset
    train_dataset = TabularDataset(mode='train', dataset_name=Dataset)
    valid_dataset = TabularDataset(mode='valid', dataset_name=Dataset)
    test_dataset = TabularDataset(mode='test', dataset_name=Dataset)

    # Get samples_per_cls for training set
    samples_per_cls = train_dataset.get_samples_per_cls()
    class_counts_dict = train_dataset.class_counts_dict

    # Create DataLoader
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True, drop_last=False)  # Drop last incomplete batch. Shuffle for training set
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size,shuffle=True)  # No shuffle for validation set
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size, shuffle=True)  # No shuffle for test set

    return train_loader, valid_loader, test_loader, samples_per_cls, class_counts_dict


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