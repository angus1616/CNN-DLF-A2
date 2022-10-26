import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision

#the foundation of this code is built off Computer Vision Assignment 4, 2022
#I have adapted the code to suit this specific task as well as created some new functionality


# top k accuracy
def topk_accuracy(output, target, topk=(1,), return_preds_correct=False):
    with torch.no_grad():
        maxk = 3
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)
        preds_topk = correct[:maxk].sum(dim=0)
        correct_3 = correct[:3].reshape(-1).float().sum(0, keepdim=True)

        if return_preds_correct:
            return (correct_3.mul_(1.0 / batch_size), preds_topk.cpu().numpy())
        else:
            return correct_3.mul_(1.0 / batch_size)


# Function to get appropriate optimiser
def get_optimiser(optimiser: str = "Adam") -> torch.optim:

    if optimiser == "Adam":
        optimiser = torch.optim.Adam
    elif optimiser == "AdamW":
        optimiser = torch.optim.AdamW
    elif optimiser == "Adamax":
        optimiser = torch.optim.Adamax
    elif optimiser == "SGD":
        optimiser = torch.optim.SGD
    elif optimiser == "SparseAdam":
        optimiser = torch.optim.SparseAdam
    elif optimiser == "Nadam":
        optimiser = torch.optim.Nadam
    else:
        raise Exception(
            "Unknown optimiser. Adam/AdamW/Adamax/SGD/SparseAdam/Adadelta")
    return optimiser


# Function to get activation
def act_func(activation: str) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=True)
    elif activation == "selu":
        return nn.SELU(inplace=True)
    elif activation == "elu":
        return nn.ELU(inplace=True)
    elif activation == "silu":
        return nn.SiLU(inplace=True)
    else:
        raise Exception("Invalid activation function")


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return None


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dl)

# plot losses


def loss_curves(history: list,
                caption: str = "Loss Curves",
                experiment_no: int = 1,
                lr: float = 0.001,
                epochs: int = 50,
                network: str = "",
                ) -> None:
    x = [instance['train_loss'] for instance in history]
    y = [instance['val_loss'] for instance in history]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(x, label="Training loss")
    ax.plot(y, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"Experiment no. {experiment_no}, {caption}: lr={lr}\nEpochs={epochs}, network type={network}")
    plt.legend()
    plt.show()

# plot accuracy
def accuracy_curves(history: list,
                  caption: str = "Accuracy Curves",
                  experiment_no: int = 1,
                  lr: float = 0.001,
                  epochs: int = 50,
                  network: str = "",
                  ) -> None:
    accuracy = [instance['val_top3_acc'] for instance in history]
    plt.plot(accuracy, label="Validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(
        f"Experiment no. {experiment_no}, {caption}: lr={lr}\nEpochs={epochs}, network type={network}")
    plt.show()

#function to display 3 fail cases 
def show_failcase(testset, testhistory):
  ar = []
  for k,v in testhistory[1][2].items():
    if len(v) > 0:
      ar.append(k)
  three_failures = ar[:3]
  fig, ax = plt.subplots(1,3,figsize=(10,15))
  plt.suptitle('Fail Cases', y=.63, fontsize=22)
  c = 0
  for index in three_failures:
    img, label = testset[index]
    ax[c].set_yticks([])
    ax[c].set_xticks([])
    ax[c].imshow(torchvision.utils.make_grid(img, nrow=1).permute(1,2,0))
    ax[c].set_title(f'Label: {label}: {testset.classes[label]}')
    c += 1
