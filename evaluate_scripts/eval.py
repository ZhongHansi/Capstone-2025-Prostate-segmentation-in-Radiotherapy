import torch
import matplotlib.pyplot as plt
from medpy.metric.binary import hd95
from SimpleITK import STAPLE
import os
from tqdm import tqdm

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

def target_plot(pred,target,img):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pred.numpy(), cmap="gray")
    plt.title("Predicted Mask")

    plt.subplot(1, 2, 2)
    plt.imshow(target.numpy(), cmap="gray")
    plt.title("Ground Truth")
    plt.show()

    plt.subplot(2, 2, 1)
    plt.imshow(img.numpy(), cmap="gray")
    plt.title("Origin Image")
    plt.show()

def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores, iou_scores, hd95_scores = [], [], []
    i = 1
    with torch.no_grad():
        for img, target in dataloader:
            img, target = img.to(device), target.to(device)

            # predict segmentation
            output = model(img, torch.tensor([0]).to(device))  
            pred = output[0].squeeze(0)
            bg = pred[1].cpu()
            bg = bg / bg.max()
            pred = pred[0].cpu()  # get the (batch, H, W)
            pred = pred / pred.max() # Normalize
            target = target.squeeze(0)
            target = torch.where(target == 2, torch.tensor(1), target) # original [0,1,2] ->[0,1]
            target = target.cpu()
            #pred = torch.sigmoid(pred)
            pred = (pred > 0.3).float()
            if i == 1:
                target_plot(pred,target,bg)
                i = 0
            #print("Unique values in pred:", torch.unique(pred))
            #print("Unique values in target:", torch.unique(target))
            # Compute scores
            dice_scores.append(dice_score(pred, target))
            iou_scores.append(iou_score(pred, target))
            hd95_scores.append(hd95(pred.numpy(), target.numpy()))

    print(f"Dice Score: {np.mean(dice_scores):.4f}")
    print(f"IoU Score: {np.mean(iou_scores):.4f}")
    print(f"HD95: {np.mean(hd95_scores):.4f}")

    return np.mean(dice_scores), np.mean(iou_scores), np.mean(hd95_scores)

def evaluate_diffusion_model(model, dataloader, device, diffusion, num_ensemble=5, threshold=0.5):
    dice_scores, iou_scores, hd95_scores = [], [], []

    with torch.no_grad():
        for image, label in dataloader:  # image: (B, C, H, W)
            image, label = image.to(device), label.to(device)

            batch_preds = []

            for _ in range(num_ensemble):
                sample_fn = diffusion.p_sample_loop_known
                sample, *_ = sample_fn(
                    model,
                    image.shape,
                    image,                # conditioning
                    step=1000,                 # diffusion steps 
                    clip_denoised=True
                )
                batch_preds.append(sample)
                #print(sample.shape)

            # Ensemble = average
            pred = torch.stack(batch_preds, dim=0).mean(0)
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()
            #print(pred.shape)
            # label process
            if label.max() == 2:
                label = torch.where(label == 2, 1, label)

            # calculate 
            pred = pred.squeeze()      # from (1,1,256,256) to (256,256)
            label = label.squeeze()
            print(pred.shape) # debug
            print(label.shape)
            dice = dice_score(pred, label)
            iou = iou_score(pred, label)
            hd = hd95(pred.cpu().numpy(), label.cpu().numpy())

            dice_scores.append(dice)
            iou_scores.append(iou)
            hd95_scores.append(hd)

    print("Final Evaluation Results:")
    print(f"Dice Score: {np.mean(dice_scores):.4f}")
    print(f"IoU Score: {np.mean(iou_scores):.4f}")
    print(f"HD95: {np.mean(hd95_scores):.4f}")

    return dice_scores, iou_scores, hd95_scores

import SimpleITK as sitk

def run_staple(preds):
    """
    preds: list of numpy arrays, each shape (H, W)
    """
    sitk_preds = [sitk.GetImageFromArray(p.astype(np.uint8)) for p in preds]
    staple_result = sitk.STAPLE(sitk_preds, 1)
    staple_result = sitk.GetArrayFromImage(staple_result)
    staple_result = (staple_result > 0.5).astype(np.float32)
    return staple_result

def visualize_prediction(pred, target, step, save_dir=None):
    """
    Visualize pred vs target, save if needed
    """
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.title('Predicted Mask')
    plt.imshow(pred, cmap='gray')
    
    plt.subplot(1,3,2)
    plt.title('Ground Truth')
    plt.imshow(target, cmap='gray')
    
    plt.subplot(1,3,3)
    plt.title('Difference')
    plt.imshow(np.abs(pred - target), cmap='hot')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/vis_step_{step}.png")
    else:
        plt.show()

    plt.close()

def evaluate_model_staple(model, dataloader, device, diffusion, num_samples=25,diffusion_step = 20,vis_freq=10, save_dir=None):
    model.eval()
    dice_scores, iou_scores, hd95_scores = [], [], []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Evaluating")

    with torch.no_grad():
        for step, (img, target) in pbar:
            img, target = img.to(device), target.to(device)
            img = 2 * img - 1   # Diffusion normalize
            # --- Diffusion sampling N times ---
            preds = []
            model_kwargs={}
            for _ in range(num_samples):
                #output = model(img, torch.tensor([0]).to(device))  
                sample, *_ = diffusion.p_sample_loop_known(
                        model,
                        img.shape,
                        img,
                        step = diffusion_step,
                        clip_denoised=True,
                        model_kwargs=model_kwargs
                    )
                pred = sample[0].squeeze(0).cpu()
                pred = pred / (pred.max() + 1e-8)    # avoid divide by 0
                pred = (pred > 0.3).float()
                preds.append(pred.numpy())

            # --- STAPLE Ensemble ---
            staple_pred = run_staple(preds)
            # --- Prepare target ---
            target = target.squeeze(0).cpu()
            target = torch.where(target == 2, torch.tensor(1), target).numpy()  # map label 2 -> 1

            # --- Metrics ---
            dice_scores.append(dice_score(torch.tensor(staple_pred[0]), torch.tensor(target)))
            iou_scores.append(iou_score(torch.tensor(staple_pred[0]), torch.tensor(target)))
            if staple_pred.sum() > 0 and target.sum() > 0:
                hd95_scores.append(hd95(staple_pred[0], target))
            else:
                hd95_scores.append(np.inf)  # no prediction or ground truth
            
            if step % vis_freq == 0:
                visualize_prediction(staple_pred[0], target, step, save_dir=save_dir)

            pbar.set_postfix({
                "Dice": f"{np.mean(dice_scores):.4f}",
                "IoU": f"{np.mean(iou_scores):.4f}",
                "HD95": f"{np.mean(hd95_scores):.2f}"
            })

    # --- Report ---
    print("="*50)
    print(f"Dice Score: {np.mean(dice_scores):.4f}")
    print(f"IoU Score: {np.mean(iou_scores):.4f}")
    print(f"HD95: {np.mean(hd95_scores):.4f}")
    print("="*50)
