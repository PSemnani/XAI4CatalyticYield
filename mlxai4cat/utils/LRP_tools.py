import torch
from typing import Any
from torch.nn.modules import Module
import torch
from torch import nn as nn
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxai4cat.utils.visualization import custom_palette
import seaborn as sns


class LRPAnalyzer:

    def __init__(self, R_svr_accumulated_all, feature_names):
        self.R_svr_accumulated_all = R_svr_accumulated_all
        self.feature_names = feature_names

    def calculate_mean_lrp_scores(self):
        self.mean_lrp_scores = np.mean(self.R_svr_accumulated_all, axis=0)
        self.df_lrp = pd.DataFrame({'Feature': self.feature_names, 'Importance Score': self.mean_lrp_scores})
        self.df_lrp_sorted = self.df_lrp.sort_values(by='Importance Score', ascending=True)

    def calculate_mean_abs_lrp_scores(self):
        self.mean_abs_lrp_scores = np.mean(np.abs(self.R_svr_accumulated_all), axis=0)
        self.df_abs_lrp = pd.DataFrame({'Feature': self.feature_names, 'Importance Score': self.mean_abs_lrp_scores})
        self.df_abs_lrp_sorted = self.df_abs_lrp.sort_values(by='Importance Score', ascending=True)

    def plot_lrp_scores(self, save_path=None):
        plt.figure(figsize=(8, 16))
        palette = custom_palette(self.df_lrp_sorted['Importance Score'].values)
        sns.barplot(x='Importance Score', y='Feature', data=self.df_lrp_sorted, palette=palette)
        plt.xlabel('Mean LRP Score', fontsize=22)
        plt.ylabel('Feature', fontsize=22)
        #plt.title('Mean LRP Scores for neuralized SVM')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def plot_abs_lrp_scores(self, save_path=None):
        plt.figure(figsize=(8, 16))
        palette = custom_palette(self.df_abs_lrp_sorted['Importance Score'].values)
        sns.barplot(x='Importance Score', y='Feature', data=self.df_abs_lrp_sorted, palette=palette)
        plt.xlabel('Mean Absolute LRP Score', fontsize=22)
        plt.ylabel('Feature', fontsize=22)
        #plt.title('Mean Absolute LRP Scores for neuralized SVM')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    def save_scores_to_csv(self, mean_lrp_filepath, mean_abs_lrp_filepath):
        self.df_lrp_sorted.to_csv(mean_lrp_filepath, index=False)
        self.df_abs_lrp_sorted.to_csv(mean_abs_lrp_filepath, index=False)


class ModifiedLinear(Module):
    def __init__(
            self,
            fc: torch.nn.Linear,
            transform: Any,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.Linear explainable.
        -------------------

        :param fc: a fully-connected layer (torch.nn.Linear).
        :param transform: a transformation function to modify the layer's parameters.
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedLinear, self).__init__()
        self.fc = fc

        if zero_bias:
            self.fc.bias = torch.nn.Parameter(torch.zeros(self.fc.bias.shape))

        self.transform = transform
        self.modified_fc = modified_layer(layer=fc, transform=transform)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        z = self.fc(x)
        zp = self.modified_fc(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()


class ModifiedLinearDiff(Module):
    def __init__(
            self,
            fc: torch.nn.Linear,
            transform: Any,
            zero_bias: bool = False
    ):
        """
        A wrapper to make torch.nn.Linear explainable.
        -------------------

        :param fc: a fully-connected layer (torch.nn.Linear).
        :param transform: a transformation function to modify the layer's parameters.
        :param zero_bias: set the layer's bias to zero. It is useful when checking the conservation property.
        """
        super(ModifiedLinearDiff, self).__init__()
        self.fc = fc

        self.fc_diff = copy.deepcopy(fc)
        self.fc_diff.weight = torch.nn.Parameter(self.fc_diff.weight - self.fc_diff.weight[[1, 0]])
        if self.fc.bias is not None:
            self.fc_diff.bias = torch.nn.Parameter(torch.zeros(self.fc.bias.shape))

        self.transform = transform
        self.modified_fc = modified_layer(layer=self.fc_diff, transform=transform)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        z = self.fc(x)
        zp = self.modified_fc(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()


class ModifiedAct(Module):
    def __init__(
            self,
            act: Any
    ):
        """
       A wrapper to make activation layers such as torch.nn.Tanh or torch.nn.ReLU explainable.
       -------------------

       :param act: an activation layer (torch.nn.Tanh or torch.nn.ReLU).
       """
        super(ModifiedAct, self).__init__()
        self.modified_act = nn.Identity()
        self.act = act

    def forward(
            self,
            x
    ):
        z = self.act(x)
        zp = self.modified_act(x)
        zp = stabilize(zp)
        return (zp.double() * (z.double() / zp.double()).data.double()).float()


class ModifiedLayerNorm(Module):
    def __init__(
            self,
            norm_layer: torch.nn.LayerNorm,
            eps: float = 1e-12,
            zero_bias: bool = False
    ):
        super(ModifiedLayerNorm, self).__init__()
        if zero_bias:
            norm_layer.bias = torch.nn.Parameter(torch.zeros(norm_layer.bias.shape))

        self.norm_layer = norm_layer
        self.weight = norm_layer.weight
        self.bias = norm_layer.bias
        self.eps = eps

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:

        z = self.norm_layer(input)
        mean = input.mean(dim=-1, keepdim=True)
        var = torch.var(input, unbiased=False, dim=-1, keepdim=True)
        denominator = torch.sqrt(var + self.eps)
        denominator = denominator.detach()
        zp = ((input - mean) / denominator) * self.weight + self.bias
        zp = stabilize(zp)
        return zp * (z / zp).data


def stabilize(z):
    """ Helper function to ensure numerical stability for LRP. """
    return z + ((z == 0.).to(z) + z.sign()) * 1e-6


def gamma(
        gam: float = 0.2
) -> torch.Tensor:
    """
    Gamma rule for LRP.
    -----------------
    :param gam: the gamma value used for propagation.
    :return: parameters modified using the gamma rule.
    """
    def modify_parameters(parameters: torch.Tensor):
        return parameters + (gam * parameters.clamp(min=0))

    return modify_parameters


def modified_layer(
        layer,
        transform
):
    """
    This function creates a copy of a layer and modify
    its parameters based on a transformation function 'transform'.
    -------------------

    :param layer: a layer which its parameters are going to be transformed.
    :param transform: a transformation function.
    :return: a new layer with modified parameters.
    """
    new_layer = copy.deepcopy(layer)

    try:
        new_layer.weight = torch.nn.Parameter(transform(layer.weight.float()))
    except AttributeError as e:
        print(e)

    try:
        new_layer.bias = torch.nn.Parameter(transform(layer.bias.float()))
    except AttributeError as e:
        print(e)

    return new_layer
