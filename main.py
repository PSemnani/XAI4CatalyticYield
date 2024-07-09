import torchvision
import torchvision.transforms as transforms
from sklearn.svm import SVC as SVC
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import pandas as pd
from neuralised_svm import neuralised_svm
#from neuralised_svm import explain
#from neuralised_opt import rbf_svm_dual
#import neuralised_opt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import os
from skopt.space import Real, Integer, Categorical
import seaborn as sns
from matplotlib.cm import ScalarMappable
from skopt import gp_minimize, space
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score


current_dir = os.getcwd()
print("Current working directory:", current_dir)

# Specify the full path to the file
filename = '../data/OCM_data/df_tree.csv'

# Check if the file exists
if os.path.exists(filename):
    # Read the dataset
    dataset = pd.read_csv(filename)
    dataset = dataset.drop(columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'GLa'])
    data_top = dataset.keys()
    print("Dataset loaded successfully!")
else:
    print("File not found:", filename)

dataset = pd.read_csv('df_tree.csv')
dataset = dataset.drop(columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'GLa'])
data_top = dataset.keys()
feature_names = data_top[1:-1]
X = dataset.iloc[:, 1:-1].values.astype('int')
y = dataset.iloc[:, -1].values.astype('int')


#removed the rs parameter
def stratified_sampling (X_pos, X_neg, y_pos, y_neg):
    X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(X_pos, y_pos, test_size = 0.2)
    X_neg_train, X_neg_test, y_neg_train, y_neg_test = train_test_split(X_neg, y_neg, test_size = 0.2)
    X_train = np.concatenate((X_pos_train, X_neg_train), axis=0)
    y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)
    X_test = np.concatenate((X_pos_test, X_neg_test), axis=0)
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)
    return X_train, y_train, X_test, y_test


# combine SMOTE and random undersampling
def resampling (X, y, overratio, underratio):
    X = X.astype(int)
    y = y.astype(int)
    over = SMOTE(sampling_strategy=overratio) # set this value to get similar sample size before and after resampling
    under = RandomUnderSampler(sampling_strategy=underratio) # to get equal sample sizes of two classes
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    # transform the dataset
    X, y = pipeline.fit_resample(X, y)
    #X = X.astype(bool)
    #y = y.astype(bool)
    return X, y

# added .astype("int")


def load_csv_and_resample(filename, overratio, underratio): 
    # Load the CSV file 
    dataset = pd.read_csv('df_tree.csv')
    dataset = dataset.drop(columns=['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', 'GLa'])
    data_top = dataset.keys()
    feature_names = data_top[1:-1]  # Assuming the first column is non-numeric and the last column is the label
    X = dataset.iloc[:, 1:-1].values  # Only select the columns that should be numeric
    y = dataset.iloc[:, -1].values  # The label column is assumed to be the last column

    # Ensure y is in integer format for the following operations
    y = y.astype(int)
    
    # Split dataset based on the label
    dataset_pos = dataset[dataset.iloc[:, -1] == 1]
    dataset_neg = dataset[dataset.iloc[:, -1] == 0]
    
    X_pos = dataset_pos.iloc[:, 1:-1].values.astype('int')
    y_pos = dataset_pos.iloc[:, -1].values.astype('int')
    X_neg = dataset_neg.iloc[:, 1:-1].values.astype('int')
    y_neg = dataset_neg.iloc[:, -1].values.astype('int')
     
    # Perform stratified sampling 
    X_train, y_train, X_test, y_test = stratified_sampling(X_pos, X_neg, y_pos, y_neg) 
     
    # Perform resampling 
    X_train, y_train = resampling(X_train, y_train, overratio=overratio, underratio=underratio) 
     
    return X_train, y_train, X_test, y_test, feature_names

X_train, y_train, X_test, y_test, feature_names = load_csv_and_resample(filename, overratio=0.6, underratio=1)


def g_of_x_from_svm(alphas, gamma, X):
    positive_alphas = alphas[alphas > 0]
    positive_sv = X_train[alphas > 0]
    negative_alphas = np.abs(alphas[alphas < 0])
    negative_sv = X_train[alphas < 0]

    pos_sample_sv_diffs = X[:, None] - positive_sv[None]
    pos_sample_sv_diffs_norms = (pos_sample_sv_diffs ** 2).sum(2)

    neg_sample_sv_diffs = X[:, None] - negative_sv[None]
    neg_sample_sv_diffs_norms = (neg_sample_sv_diffs ** 2).sum(2)

    g = np.log(np.exp(-gamma * pos_sample_sv_diffs_norms) @ positive_alphas) - np.log(
        np.exp(-gamma * neg_sample_sv_diffs_norms) @ negative_alphas)
    return g

dataset_pos = dataset.iloc[y==True,].values.astype('bool')
dataset_neg = dataset.iloc[y==False,].values.astype('bool')
X_pos = dataset_pos[:, 1:-1]
y_pos = dataset_pos[:, -1]
X_neg = dataset_neg[:, 1:-1]
y_neg = dataset_neg[:, -1]

def custom_palette(arr):
    """Create a custom color palette based on the relevance scores."""
    min_val = min(arr)
    max_val = max(arr)
    n_colors = len(arr) * 2

    if min_val >= 0:  # Only red part
        return list(sns.color_palette("coolwarm", n_colors=n_colors)[len(arr):])
    elif max_val <= 0:  # Only blue part
        return list(sns.color_palette("coolwarm", n_colors=n_colors)[:len(arr)])
    else:
        num_negative = len([x for x in arr if x <= 0])
        num_positive = len(arr) - num_negative

        palette = []
        if num_negative > num_positive:
            # More negative values, assign blue to more portion
            blue_part = int(num_negative / len(arr) * n_colors)
            palette.extend(sns.color_palette("coolwarm", n_colors=blue_part)[:num_negative])
            palette.extend(sns.color_palette("coolwarm", n_colors=n_colors - blue_part)[len(arr) - num_negative:])
        else:
            # More positive values, assign red to more portion
            red_part = int(num_positive / len(arr) * n_colors)
            palette.extend(sns.color_palette("coolwarm", n_colors=n_colors - red_part)[:len(arr) - num_positive])
            palette.extend(sns.color_palette("coolwarm", n_colors=red_part)[num_positive:])

        return palette

class LRPAnalyzer:
    def __init__(self, R_svr_accumulated_all, feature_names):
        self.R_svr_accumulated_all = R_svr_accumulated_all
        self.feature_names = feature_names
    
    def calculate_mean_lrp_scores(self):
        self.mean_lrp_scores = np.mean(self.R_svr_accumulated_all, axis=0)
        self.df_lrp = pd.DataFrame({'Feature': self.feature_names, 'Mean_LRP_Score': self.mean_lrp_scores})
        self.df_lrp_sorted = self.df_lrp.sort_values(by='Mean_LRP_Score', ascending=True)
    
    def calculate_mean_abs_lrp_scores(self):
        self.mean_abs_lrp_scores = np.mean(np.abs(self.R_svr_accumulated_all), axis=0)
        self.df_abs_lrp = pd.DataFrame({'Feature': self.feature_names, 'Mean_Absolute_LRP_Score': self.mean_abs_lrp_scores})
        self.df_abs_lrp_sorted = self.df_abs_lrp.sort_values(by='Mean_Absolute_LRP_Score', ascending=True)

    def custom_palette(self, arr):
        """Create a custom color palette based on the relevance scores."""
        min_val = min(arr)
        max_val = max(arr)
        n_colors = len(arr) * 2

        if min_val >= 0:  # Only red part
            return list(sns.color_palette("coolwarm", n_colors=n_colors)[len(arr):])
        elif max_val <= 0:  # Only blue part
            return list(sns.color_palette("coolwarm", n_colors=n_colors)[:len(arr)])
        else:
            num_negative = len([x for x in arr if x <= 0])
            num_positive = len(arr) - num_negative

            palette = []
            if num_negative > num_positive:
                # More negative values, assign blue to more portion
                blue_part = int(num_negative / len(arr) * n_colors)
                palette.extend(sns.color_palette("coolwarm", n_colors=blue_part)[:num_negative])
                palette.extend(sns.color_palette("coolwarm", n_colors=n_colors - blue_part)[len(arr) - num_negative:])
            else:
                # More positive values, assign red to more portion
                red_part = int(num_positive / len(arr) * n_colors)
                palette.extend(sns.color_palette("coolwarm", n_colors=n_colors - red_part)[:len(arr) - num_positive])
                palette.extend(sns.color_palette("coolwarm", n_colors=red_part)[num_positive:])

            return palette
    
    def plot_lrp_scores(self, save_path=None):
        plt.figure(figsize=(8, 16))
        palette = self.custom_palette(self.df_lrp_sorted['Mean_LRP_Score'].values)
        sns.barplot(x='Mean_LRP_Score', y='Feature', data=self.df_lrp_sorted, palette=palette)
        plt.xlabel('Importance Score', fontsize=22)
        plt.ylabel('Feature', fontsize=22)
        #plt.title('Mean LRP Scores for neuralized SVM')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_abs_lrp_scores(self, save_path=None):
        plt.figure(figsize=(8, 16))
        palette = self.custom_palette(self.df_abs_lrp_sorted['Mean_Absolute_LRP_Score'].values)
        sns.barplot(x='Mean_Absolute_LRP_Score', y='Feature', data=self.df_abs_lrp_sorted, palette=palette)
        plt.xlabel('Importance Score', fontsize=22)
        plt.ylabel('Feature', fontsize=22)
        #plt.title('Mean Absolute LRP Scores for neuralized SVM')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def save_scores_to_csv(self, mean_lrp_filepath, mean_abs_lrp_filepath):
        self.df_lrp.to_csv(mean_lrp_filepath, index=False)
        self.df_abs_lrp.to_csv(mean_abs_lrp_filepath, index=False)


if __name__ == '__main__':
    # Define lists to store feature importance values for each split
    R_svr_accumulated_all = []

    # Perform N train-test splits
    N = 100
    #f1_scores = []
    #accuracies = []
    acc_svm = []
    precision_svm = []
    recall_svm = []
    f1_svm = []
    for rs in range(N):
        if rs % 20 == 0:
            print(f'train/test split {rs}')
        X_train, y_train, X_test, y_test = stratified_sampling(X_pos, X_neg, y_pos, y_neg)
        X_train, y_train = resampling(X_train, y_train, overratio=0.6, underratio=1)

        # X_test, y_test = resampling(X_test, y_test, overratio=0.6, underratio=1)
        #print('y test shape', y_test.shape)
        #print('y test pos', np.sum(y_test > 0))

        C_optimized = 22 #100
        gamma_optimized = 0.01

        # Train an SVM model with the optimized hyperparameters
        svm = SVC(kernel='rbf', gamma=gamma_optimized, C=C_optimized)
        svm.fit(X_train, y_train)
        svm_pred = svm.predict(X_test)
        acc_svm.append(accuracy_score(y_test, svm_pred))
        precision_svm.append(precision_score(y_test, svm_pred, zero_division=1))
        recall_svm.append(recall_score(y_test, svm_pred))
        f1_svm.append(f1_score(y_test, svm_pred))
        print(f' mean f1 score: {np.mean(f1_svm)}')
        print('accuracy', np.mean(acc_svm))
        #f1 = f1_score(y_test, svm_pred)
        #acc = svm.score(X_test, y_test)
        #accuracies.append(acc)
        #f1_scores.append(f1)

        # Perform neuralisation
        svm_neural = neuralised_svm(svm)
        R_svr_accumulated = svm_neural.explain(X_test, first_rule="GI", with_intercept=False, reweight_explanation=True)
        R_svr_accumulated_all.append(R_svr_accumulated)
        svm_pred = svm.predict(X_test)
        print('svm pred pos', np.sum(svm_pred > 0))
        #alphas = np.zeros_like(y_train).astype(float)
        #alphas[svm.support_] = svm.dual_coef_[0]
        #neuralised_svm = rbf_svm_dual(np.array(X_train), np.array(y_train), np.array(alphas), gamma_optimized)
        # Calculate LRP scores for each sample in the test set
        #R_svr_accumulated = []
        #for sample in X_test:
        #    R_svr = neuralised_svm.lrp(sample, svm, last_propagation="svr")[0]
        #    R_svr_accumulated.append(R_svr)
        # Append the accumulated LRP scores to the list ABSOLUTE VALUE
        #R_svr_accumulated_all.append(R_svr_accumulated)

    analyzer = LRPAnalyzer(np.array(R_svr_accumulated_all).reshape(-1, len(feature_names)), feature_names)
    analyzer.calculate_mean_lrp_scores()
    analyzer.calculate_mean_abs_lrp_scores()
    analyzer.plot_lrp_scores()
    analyzer.plot_abs_lrp_scores()
    analyzer.plot_lrp_scores('/Users/parastoo/phd_projects/OCM/Explaining_SVM/results/sorted_mean_lrp_SVM_GI.png')
    analyzer.plot_abs_lrp_scores('/Users/parastoo/phd_projects/OCM/Explaining_SVM/results/sorted_mean_abs_lrp_SVM_GI.png')
    analyzer.save_scores_to_csv('/Users/parastoo/phd_projects/OCM/Explaining_SVM/results/sorted_mean_lrp_SVM.csv', '/Users/parastoo/phd_projects/OCM/Explaining_SVM/results/sorted_mean_abs_lrp_SVM.csv')

    ratio_pos2neg = []
    for R_svr_accumulated in R_svr_accumulated_all:
        split_array = R_svr_accumulated
        pos_element = np.sum(split_array[split_array > 0])
        neg_element = np.sum(np.abs(split_array[split_array < 0]))
        ratio = pos_element / neg_element
        #print(f"pos/neg: {ratio}")
        ratio_pos2neg.append(ratio)

mean_ratio = np.mean(ratio_pos2neg)
print("ratio mean: ", mean_ratio)




