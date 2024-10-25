import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pandas as pd
import torch


def prepare_dataset(data_path):
    dataset = pd.read_csv(data_path)
    #Remove periodic system group info as it is highly correlated with the other elemnts features
    dataset = dataset.drop(columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'GLa'])
    data_top = dataset.keys()
    feature_names = data_top[1:-1]
    X = dataset.iloc[:, 1:-1].values.astype('int')
    y = dataset.iloc[:, -1].values.astype('int')
    dataset_pos = dataset.iloc[y == True,].values.astype('bool')
    dataset_neg = dataset.iloc[y == False,].values.astype('bool')
    X_pos = dataset_pos[:, 1:-1]
    y_pos = dataset_pos[:, -1]
    X_neg = dataset_neg[:, 1:-1]
    y_neg = dataset_neg[:, -1]

    return dataset, X, y, X_pos, y_pos, X_neg, y_neg, feature_names


def stratified_sampling (X_pos, X_neg, y_pos, y_neg, rs):
    X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(X_pos, y_pos, test_size = 0.2, random_state = rs)
    X_neg_train, X_neg_test, y_neg_train, y_neg_test = train_test_split(X_neg, y_neg, test_size = 0.2, random_state = rs)
    X_train = np.concatenate((X_pos_train, X_neg_train), axis=0)
    y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)
    X_test = np.concatenate((X_pos_test, X_neg_test), axis=0)
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)
    return X_train, y_train, X_test, y_test


def resampling(X, y, overratio, underratio, randomstate):
    # combine SMOTE and random undersampling
    X = X.astype(int)
    y = y.astype(int)
    over = SMOTE(sampling_strategy=overratio, random_state=randomstate) # set this value to get similar sample size before and after resampling
    under = RandomUnderSampler(sampling_strategy=underratio, random_state=randomstate) # to get equal sample sizes of two classes
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)
    X = X.astype(bool)
    y = y.astype(bool)
    return X, y


class CatalystDataset(torch.utils.data.Dataset):
    """A data class for the catalyst dataset to be used with PyTorch."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        data = self.X[index]
        label = self.y[index]

        return data, label

    def __len__(self):
        return len(self.X)


def get_test_data_loader(X_pos, X_neg, y_pos, y_neg, rs):
    """This method creates the test data loaders for a given split (rs), and the returns the remaining trainig data. If resapling is set to False, the data is not resampled."""

    X_train, y_train, X_test, y_test = stratified_sampling(X_pos, X_neg, y_pos, y_neg, rs)

    # shuffle train set
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    catalyst_dataset_test = CatalystDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(catalyst_dataset_test, shuffle=False)

    return X_train, y_train, test_loader


def get_xval_data_loaders(k, k_i, X_train, y_train, with_resampling=True, verbose=False):
    """This method creates the train and validation loaders for a cross-validation run, splitting the data based on the number of k and the current cross-validation run.
    If resapling is set to False, the data is not resampled."""
    split_idx = np.linspace(0, X_train.shape[0], num=(k + 1)).astype(int)
    X_train_split = [X_train[split_idx[i]:split_idx[i + 1]] for i in range(k)]
    y_train_split = [y_train[split_idx[i]:split_idx[i + 1]] for i in range(k)]
    X_val = X_train_split[k_i]
    y_val = y_train_split[k_i]
    X_train_new = np.concatenate([X_train_split[i] for i in range(k) if i != k_i], axis=0)
    y_train_new = np.concatenate([y_train_split[i] for i in range(k) if i != k_i], axis=0)
    if with_resampling:
        X_train_new, y_train_new = resampling(X_train_new, y_train_new, overratio=0.6, underratio=1, randomstate=123)
    if verbose:
        print(f"Train data shape: {X_train_new.shape}, {y_train_new.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    catalyst_dataset_train = CatalystDataset(X_train_new, y_train_new)
    catalyst_dataset_val = CatalystDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(catalyst_dataset_train)
    val_loader = torch.utils.data.DataLoader(catalyst_dataset_val)

    return train_loader, val_loader
