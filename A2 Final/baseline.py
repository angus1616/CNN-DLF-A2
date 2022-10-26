
from torch._C import _get_default_device
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from utils import *
import numpy as np
import time

#the foundation of this code is built off Computer Vision Assignment 4, 2022
#I have adapted the code to suit this specific task as well as created some new functionality
class Base(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self,
                        batch: list,
                        test: bool = False,
                        return_preds: bool = False,
                        ) -> dict:
        # batch
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        top3_acc = topk_accuracy(out, labels, (3), return_preds)
        if return_preds:
          fail_cases = []
          failures = np.where(top3_acc[1] == 0)[0]
          for i in failures:
            fail_cases.append((labels[i].item(), images[i].cpu().numpy()))
        if test:
          if return_preds:
            return {"test_loss": loss.detach(), "test_top3_acc": top3_acc[0]}, {"correct_predictions": top3_acc[1], "labels": labels.cpu().numpy(), "fail_cases": fail_cases}
          else:
            return {"test_loss": loss.detach(), "test_top3_acc": top3_acc[0]}
        else:
            return {"val_loss": loss.detach(), "val_top3_acc": top3_acc[0]}
  
    def validation_epoch_end(self,
                             outputs: dict,
                             test: bool = False):
  
        if test:
            batch_losses = [x['test_loss'] for x in outputs]
            batch_accs = [x['test_top3_acc'] for x in outputs]
        else:
            batch_losses = [x['val_loss'] for x in outputs]
            batch_accs = [x['val_top3_acc'] for x in outputs]
        # stack to combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        # same for accuracies
        epoch_acc = torch.stack(batch_accs).mean()
        return {'test_loss': epoch_loss.item(), 'test_top3_acc': epoch_acc.item()} if test else {'val_loss': epoch_loss.item(), 'val_top3_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        # print every five epochs
        if (epoch) % 5 == 0:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_top3_acc']))


@torch.no_grad()
def evaluate(model: nn.Module,
             val_loader: torch.utils.data.DataLoader,
             test: bool = False,
             val_transforms: torchvision.transforms = None,
             return_preds: bool = False) -> None:

    model.eval()
    outputs = []
    if val_transforms is not None:
      val_loader.dl.dataset.transform = val_transforms
    if return_preds:
      predictions = []
      target_labels = []
      #dictionary of fail indexes and images
      fail_cases = {i: [] for i in range(0,101)}
    for b in val_loader:
      #condition to return predictions (correct and failure)
      if return_preds:
          output, predictions_val = model.validation_step(b, test, return_preds)
      else:
          output = model.validation_step(b, test, return_preds)
      outputs.append(output)
      if return_preds:
        predictions.extend(predictions_val['correct_predictions'])
        target_labels.extend(predictions_val['labels'])
        for i in predictions_val['fail_cases']:
          #append fails to empty array in fail_cases
          fail_cases[i[0]].append(i[1])
    #third condition check to return outputs
      if return_preds:
        return model.validation_epoch_end(outputs, test = test), (predictions, target_labels, fail_cases)
      else:
        return model.validation_epoch_end(outputs, test = test)


def fit(model: torch.nn.Module,
        train_loader: torch.utils.data.dataloader,
        val_loader: torch.utils.data.dataloader,
        epochs: int = 50,
        momentum: float = 0.9,
        weight_decay: float = 0,
        lr: float = 0.001,
        lr_scheduler: str = None,
        opt_func: str = 'SGD',
        train_transforms: torchvision.transforms = None,
        val_transforms: torchvision.transforms = None,
        return_preds: bool = False,
        ):

    torch.cuda.empty_cache()
    begin = time.time()
    history = []
    if opt_func == 'SGD':
        kw_args = {"lr": lr, "momentum": momentum,
                   "weight_decay": weight_decay}
        optimiser = get_optimiser(opt_func)(
            model.parameters(), **kw_args)

    elif opt_func == 'Adabound':
        optimiser = optim.AdaBound(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            final_lr=0.1,
            gamma=1e-3,
            eps=1e-8,
            weight_decay=0,
            amsbound=False,
        )
    else:
        optimiser = get_optimiser(opt_func)(
            model.parameters())

    # can be onecycle or ROPlateau
    if lr_scheduler == 'OneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser,
                                                        max_lr=.01,
                                                        total_steps=epochs *
                                                        (len(train_loader)),
                                                        steps_per_epoch=len(
                                                            train_loader)
                                                        )
    elif lr_scheduler == 'ROPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min",
            patience=3,
            cooldown=2,
            verbose=True, factor=0.5,
            min_lr=1e-6)
    else:
        lr_scheduler = None

    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        # set unique train transforms based on model
        if train_transforms is not None:
          #update transformations
          #remember we double wrapped into devicedataloader
            train_loader.dl.dataset.dataset.transform = train_transforms
        for i, batch in enumerate(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            if lr_scheduler == 'OneCycle':
                scheduler.step()
            # Validation phase
        result = evaluate(model, val_loader, val_transforms=val_transforms, return_preds=return_preds)
        if lr_scheduler == 'ROPlateau':
            scheduler.step(result['val_loss'])
        if result['val_loss'] < best_val_loss:
            best_val_loss = result['val_loss']
            best_val_acc = result['val_top3_acc']
            model_best_model_state_dict = model.state_dict()
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
        #save models for testing
        model.best_model_path = f'model:{model.model_name}, Run:{model.model_run_no}.pt'
        torch.save(model_best_model_state_dict, model.best_model_path)
        finish = time.time()
    print(f'Training Duration: {(finish-begin)/60:.2f} minutes.')
    return history