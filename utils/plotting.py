import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os


def get_formatted_results(acc, f1, precision, recall, model_name, verbose=True, df_metrics=None):
    mean_acc, std_acc = np.mean(acc), np.std(acc)  # mean and std over different data splits
    mean_f1, std_f1 = np.mean(f1), np.std(f1)
    mean_precision, std_precision = np.mean(precision), np.std(precision)
    mean_recall, std_recall = np.mean(recall), np.std(recall)

    if verbose:
        table_data_mlp = [
            ('Metric', 'Mean', 'Standard Deviation'),
            ('Accuracy', mean_acc, std_acc),
            ('F1 Score', mean_f1, std_f1),
            ('Precision', mean_precision, std_precision),
            ('Recall', mean_recall, std_recall)
        ]
        print(tabulate(table_data_mlp, headers='firstrow', tablefmt='fancy_grid'))

    data = {
        'Model': model_name,
        'Accuracy_Mean': mean_acc,
        'Accuracy_Std': std_acc,
        'F1_Mean': mean_f1,
        'F1_Std': std_f1,
        'Precision_Mean': mean_precision,
        'Precision_Std': std_precision,
        'Recall_Mean': mean_recall,
        'Recall_Std': std_recall
    }

    # Convert data dictionary to DataFrame
    new_row = pd.DataFrame([data])

    # Check if the DataFrame exists. If not, create one with the new data
    if df_metrics is None:
        df_metrics = new_row
    else:
        # Append the new data to the existing DataFrame
        df_metrics = pd.concat([df_metrics, new_row], ignore_index=True)

    return df_metrics


def plot_feature_importance(feature_importances, feature_names, model_name, df_feature_importance=None, savedir='figures'):
    """
    This function updates an existing DataFrame of feature importances or creates a new one,
    then sorts and plots the feature importances for a given model.
    The values are normalized to the range [0, 1].

    Parameters:
    - feature_importances (array-like): The importance values of the features.
    - feature_names (list): List of feature names.
    - model_name (str): Name of the model, to be used as a column header and part of file names.
    - df_feature_importance (pd.DataFrame, optional): Existing DataFrame of feature importances. If None, a new one will be created.

    Returns:
    - df_feature_importance (pd.DataFrame): Updated DataFrame with the new model's feature importances.
    """

    # Check if the DataFrame exists. If not, create one
    if df_feature_importance is None:
        df_feature_importance = pd.DataFrame({'Feature': feature_names})

    # Calculate the mean of the feature importances
    feature_importances_mean = np.mean(feature_importances, axis=0)

    # Normalize the feature importances to the range [0, 1]
    normalized_importances = (feature_importances_mean - feature_importances_mean.min()) / (feature_importances_mean.max() - feature_importances_mean.min())

    df_feature_importance[model_name] = normalized_importances

    # Create a new DataFrame for plotting
    df_fi_model = df_feature_importance[['Feature', model_name]].copy()
    df_fi_model = df_fi_model.sort_values(by=model_name, ascending=False)

    # Normalize and map colors based on the importance values
    min_val = min(df_fi_model[model_name].min(), 0)
    max_val = df_fi_model[model_name].max()
    norm = plt.Normalize(min_val, max_val)
    colors = plt.cm.copper_r(norm(df_fi_model[model_name]))

    # Plotting
    _, ax = plt.subplots(figsize=(10, 14))
    ax.barh(df_fi_model['Feature'], df_fi_model[model_name], color=colors)
    plt.title(f'Feature Importance in {model_name}')
    ax.set_xlabel('Normalized Feature Importance', fontsize=16)
    ax.set_ylabel('Features', fontsize=16)
    ax.set_yticks(np.arange(1, len(df_fi_model['Feature']) + 1))
    ax.set_yticklabels(df_fi_model['Feature'], fontsize=12)
   # ax.set_xticklabels([0, 0.2, 0.4, 0,6, 0.8, 1], fontweight='bold')
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    plt.savefig(os.path.join(savedir, 'FI_{model_name}.png'))
    plt.show()

    return df_feature_importance
