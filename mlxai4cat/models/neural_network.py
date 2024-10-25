import torch
from typing import Any
from torch.nn.modules import Module
from mlxai4cat.utils import LRP_tools as LRP
import torch
from torch import nn as nn
import copy
from mlxai4cat.utils.LRP_tools import gamma, stabilize
from tqdm import tqdm
from collections import defaultdict


class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network class."""
    def __init__(
            self,
            num_layers=3,
            num_neurons_per_layer=[36, 128, 128, 2],
            dropout_rate=0,
            bias=True,
            layer_norm=False,
            output_tanh=False,
            artificial_neuron=False,

    ):
        """
        :param num_layers: number of layers in the network.
        :param num_neurons_per_layer: number of neurons in each layer.
        :param dropout_rate: dropout rate.
        :param bias: if True, use bias in the layers.
        :param layer_norm: if True, apply layer normalization to the layers.
        :param output_tanh: if True, apply tanh activation to the output layer.
        :param artificial_neuron: if True, use an artificial neuron in the layers for relevance propagation.
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.output_tanh = None
        self.artificial_neuron = artificial_neuron

        layers = []
        layer_norms = []
        for i in range(self.num_layers - 1):
            layers.append(nn.Linear(self.num_neurons_per_layer[i], self.num_neurons_per_layer[i + 1], bias=bias))
            if self.layer_norm:
                layer_norms.append(nn.LayerNorm(self.num_neurons_per_layer[i + 1]))

        if self.artificial_neuron:
            layers.append(nn.Linear(self.num_neurons_per_layer[-2], self.num_neurons_per_layer[-1], bias=False))
        else:
            layers.append(nn.Linear(self.num_neurons_per_layer[-2], self.num_neurons_per_layer[-1], bias=bias))

        self.dropout = nn.Dropout(p=dropout_rate)
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(layer_norms)
        self.act = nn.ReLU()
        if output_tanh:
            self.output_tanh = nn.Tanh()

    def forward(self, x):
        out = self.dropout(x)
        for i, layer in enumerate(self.layers):
            out = layer(out)

            if i != self.num_layers - 1:
                out = self.act(out)
                if self.layer_norm:
                    out = self.layer_norms[i](out)
        if self.output_tanh is not None:
            out = self.output_tanh(out) + 1
        return out


class ModifiedNeuralNetwork(nn.Module):
    """ Wrapper class for the feedforward neural network class which modifies the layers to enable relevance propagation."""

    def __init__(self, net):
        """
        :param net: the original neural network model.
        """
        super().__init__()

        self.num_layers = net.num_layers
        self.num_neurons_per_layer = net.num_neurons_per_layer
        self.dropout_rate = net.dropout_rate
        self.layer_norm = net.layer_norm
        self.output_tanh = net.output_tanh
        self.artificial_neuron = net.artificial_neuron

        layers = []
        layer_norms = []
        for layer in net.layers[:-1]:
            layers.append(LRP.ModifiedLinear(fc=layer, transform=gamma()))
        if net.artificial_neuron:
            layers.append(LRP.ModifiedLinearDiff(fc=net.layers[-1], transform=gamma()))
        else:
            layers.append(LRP.ModifiedLinear(fc=net.layers[-1], transform=gamma()))

        for layer_norm in net.layer_norms:
            mod_ln = LRP.ModifiedLayerNorm(norm_layer=layer_norm)
            layer_norms.append(mod_ln)

        self.dropout = net.dropout
        self.layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(layer_norms)
        self.act = LRP.ModifiedAct(net.act)
        if self.output_tanh is not None:
            self.output_tanh = LRP.ModifiedAct(net.output_tanh)

    def forward(self, x):
        out = self.dropout(x)
        for i, layer in enumerate(self.layers):
            out = layer(out)

            if i != self.num_layers - 1:
                out = self.act(out)
                if self.layer_norm:
                    out = self.layer_norms[i](out)
        if self.output_tanh is not None:
            out = self.output_tanh(out) + 1
        return out

    def relevance_act(self, x):
        """
        Function to calculate the relevance activation for the input, enabling relevance rebalancing.
        --------------------------------------------------
        :param x: input vector.
        """
        if self.artificial_neuron:
            out = self.dropout(x)
            for i, layer in enumerate(self.layers[:-1]):
                out = layer(out)
                out = self.act(out)
                if self.layer_norm:
                    out = self.layer_norms[i](out)
            zp = out.unsqueeze(1) * self.layers[-1].modified_fc.weight.unsqueeze(0)
            z = out.unsqueeze(1) * self.layers[-1].fc.weight.unsqueeze(0)
            zp = stabilize(zp)
            out = (zp.double() * (z.double() / zp.double()).data.double()).float()
            return out
        else:
            return self.forward(x)

    def inference_with_relevance(self, data_loader, reweight_explanation=True,
                                 relevance_on_positive_class=True):
        """
        Perform inference including predictions, relevance and confusion matrices.
        --------------------------------------------------------------------
        :param data_loader: DataLoader containing inference data
        :param reweight_explanation: whether to reweight relevance to balance between classes
        :param relevance_on_positive_cls: if True base relevance on positive class, otherwise on predicted class

        :return pred: predicted labels
        :return gt: ground truth labels
        :return rels: relevance scores
        :return confusion_scores: classifier scores sparated in confusion categories
        :return confusion_idxs: indices of the test samples contained in each confusion category 
        """

        pred = []
        gt = []
        probs = []
        rels = []
        confusion_scores = defaultdict(list)
        confusion_idxs = defaultdict(list)

        for i, (x, y) in tqdm(enumerate(data_loader)):
            x = torch.tensor(x).to(next(self.parameters()))
            y = torch.tensor(y).long()  # Assuming y is already in the appropriate form

            logits = self(x)
            pred_probab = nn.Softmax(dim=1)(logits)
            y_pred = pred_probab.argmax(1)  # Get the predicted class

            pred.append(y_pred.item())
            probs.append(pred_probab.numpy(force=True))
            gt.append(y.item())  # Directly use y since it's already the class label

            x = x.detach().cpu()
            # Perform LRP analysis
            rel_x = self.get_relevance(x, y, reweight_explanation,
                                       relevance_on_positive_class)
            rel_x = torch.detach(rel_x).cpu()
            rels.append(rel_x)
            # Evaluate and sort scores by outcome
            if y_pred == y:
                if y_pred.item() == 0:
                    confusion_scores['true_neg_scores'].append(pred_probab[0, 0].item())
                    confusion_idxs['true_neg_idx'].append(i)
                else:
                    confusion_scores['true_pos_scores'].append(pred_probab[0, 1].item())
                    confusion_idxs['true_pos_idx'].append(i)
            else:
                if y_pred.item() == 0:
                    confusion_scores['false_neg_scores'].append(pred_probab[0, 0].item())
                    confusion_idxs['false_neg_idx'].append(i)
                else:
                    confusion_scores['false_pos_scores'].append(pred_probab[0, 1].item())
                    confusion_idxs['false_pos_idx'].append(i)
        return pred, gt, probs, rels, confusion_scores, confusion_idxs

    def get_relevance(self, X, target, reweight_explanation=True,
                      relevance_on_positive_class=True):
        """
        Apply LRP given a model, set of inputs, and a target label with respect to which relenvace will be calculated.
        --------------------------------------------------------------
        :param model: a modified neural network model
        :param X: tensor of batched inputs
        :param target: tensor of target labels

        :return relevance: relevance scores for all features of each input
        :return pred: predicted labels
        :return output: classifier output scores used for LRP
        """
        if relevance_on_positive_class:
            target = torch.tensor([[0, 1]])
        else:
            target = torch.nn.functional.one_hot(target, num_classes=2).float()  # One-hot encoding for LRP if necessary

        X = (X * 1).data
        X.requires_grad_(True)
        X.grad = None

        # Get the relevance scores.
        output = self.forward(X)

        if target is None:
            (output[..., 1] - output[..., 0]).sum().backward()
        else:
            (output * target).sum().backward()

        # (output).sum().backward()
        relevance = (X.grad * X).data

        rel_output = self.relevance_act(X)[:, 1]

        if reweight_explanation:
            relevance = self.reweight_explanation(relevance, rel_output)

        return relevance

    def reweight_explanation(self, rels, lrp_out):
        pos_out_mask = (lrp_out > 0).to(float)
        neg_out_mask = (lrp_out <= 0).to(float)
        sum_pos_out = (pos_out_mask * lrp_out).sum(-1)
        sum_neg_out = (neg_out_mask * torch.abs(lrp_out)).sum(-1)
        reweight_out = torch.stack([sum_neg_out, sum_pos_out], dim=-1)

        pos_rel_mask = (rels > 0).to(torch.float)
        neg_rel_mask = (rels <= 0).to(torch.float)
        sum_pos_rel = (pos_rel_mask * rels).sum(1, keepdims=True)
        sum_neg_rel = (neg_rel_mask * torch.abs(rels)).sum(1, keepdims=True)

        reweight_fac = torch.zeros_like(rels)

        sum_pos_fac = pos_rel_mask * (reweight_out[:, [1]] / (sum_pos_rel + 1e-10))
        sum_neg_fac = neg_rel_mask * (reweight_out[:, [0]] / (sum_neg_rel + 1e-10))
        reweight_fac += sum_pos_fac
        reweight_fac += sum_neg_fac

        reweight_rels = rels * reweight_fac

        return reweight_rels
