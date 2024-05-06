import torch
import numpy as np
import torch

def count_metric(model, test_dataloader, metric, device='cpu'):
    avg = 0
    model.eval()
    for X_batch, Y_batch in test_dataloader:
        with torch.no_grad():
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred = torch.sigmoid(model(X_batch))
            val = metric(pred, Y_batch)
            avg += val

    return avg.item() / len(test_dataloader)


def dice_coef(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(y_pred * y_true, dim=(1, 2, 3))
    union = torch.sum(y_pred, dim=(1, 2, 3)) + torch.sum(y_true, dim=(1, 2, 3))
    return torch.mean((2.0 * intersection) / (union + 1e-6))
