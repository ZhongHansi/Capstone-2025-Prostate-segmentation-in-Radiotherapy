import torch

def dice_score(pred, target, smooth=1e-6):
    """
    calculate Dice
    :param pred: predicted segmentation (batch, H, W) or (batch, 1, H, W)
    :param target: Ground truth segmentation (batch, H, W) or (batch, 1, H, W)
    :param smooth: Smoothing factor to avoid division by zero 避免除零
    :return: Dice coefficient
    """
    pred = torch.sigmoid(pred)  
    pred = (pred > 0.5).float()  # Binarize the predictions 二值化
    intersection = (pred * target).sum()  # Compute intersection
    union = pred.sum() + target.sum()  # Compute union

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()  # Return mean Dice score

def iou_score(pred, target, smooth=1e-6):
    """
    calculate IoU
    :param pred: predicted segmentation (batch, H, W)
    :param target: Ground truth segmentation (batch, H, W)
    :param smooth: Smoothing factor to avoid division by zero
    :return: IoU score
    """
    pred = torch.sigmoid(pred)  
    pred = (pred > 0.5).float()  

    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection  # union

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def hd95(pred, target):
    """
    Compute the Hausdorff 95% Distance
    :param pred: Predicted mask (numpy array, H, W)
    :param target: Ground truth mask (numpy array, H, W)
    :return: HD95 Distance
    """
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return np.inf  # Return infinity if no valid regions are found

    # compute Hausdorff Distance
    distances_1 = [directed_hausdorff(pred_points, target_points)[0] for _ in range(10)]
    distances_2 = [directed_hausdorff(target_points, pred_points)[0] for _ in range(10)]

    return np.percentile(distances_1 + distances_2, 95)  # Compute 95%

def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores, iou_scores, hd95_scores = [], [], []

    with torch.no_grad():
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)

            # predict segmentation
            output = model(img, torch.tensor([0]).to(device))  
            pred = output[0].squeeze(0)
            pred = pred[0].cpu()  # get the (batch, H, W)
            target = target.squeeze(0)
            #target = torch.where(target == 2, torch.tensor(1), target) # original [0,1,2] ->[0,1]
            target = target.cpu()
            pred = (pred > 0.5).float()
            print("Unique values in pred:", torch.unique(pred))
            print("Unique values in target:", torch.unique(target))
            # Compute scores
            dice_scores.append(dice_score(pred, target))
            iou_scores.append(iou_score(pred, target))
            hd95_scores.append(hd95(pred.numpy(), target.numpy()))

    print(f"Dice Score: {np.mean(dice_scores):.4f}")
    print(f"IoU Score: {np.mean(iou_scores):.4f}")
    print(f"HD95: {np.mean(hd95_scores):.4f}")

    return np.mean(dice_scores), np.mean(iou_scores), np.mean(hd95_scores)
