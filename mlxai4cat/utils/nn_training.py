import torch
import torch.nn as nn

def train_epoch(train_loader, model, criterion, optimizer):
    """
    Perform one epoch of training with a classification model.
    --------------------------------------------------------------------
    :param train_loader: DataLoader containing training data
    :param model: model to be trained
    :param criterion: loss function
    :param optimizer: optimizer
    """

    for _, (x_train, y_train) in enumerate(train_loader):
        x_train = torch.tensor(x_train).to(next(model.parameters()))
        y_train = torch.nn.functional.one_hot(torch.tensor(y_train).long(), 2).to(x_train)
        optimizer.zero_grad()

        logits_train = model(x_train.float())

        loss = criterion(torch.nn.Softmax()(logits_train), y_train)
        loss.backward()
        optimizer.step()

        # pred_probab_train = nn.Softmax(dim=1)(logits_train)
        # y_pred_train = pred_probab_train.argmax(1)


def val_epoch(val_loader, model):
    """
    Perform one step validation with a classification model.
    --------------------------------------------------------------------
    :param val_loader: DataLoader containing validation data
    :param model: model to be trained

    :return val_pred: predicted labels
    :return val_gt: ground truth labels
    """
    val_pred = []
    val_gt = []
    for _, (x_val, y_val) in enumerate(val_loader):
        x_val = torch.tensor(x_val).to(next(model.parameters()))
        y_val = torch.nn.functional.one_hot(torch.tensor(y_val).long(), 2).to(x_val)

        logits_val = model(x_val.float())
        pred_probab_val = torch.nn.Softmax(dim=1)(logits_val)
        y_pred_val = pred_probab_val.argmax(1)

        val_pred.append(y_pred_val.item())
        val_gt.append(y_val.argmax(1).item())
    return val_pred, val_gt


