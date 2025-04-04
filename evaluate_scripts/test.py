from .model_loader import custom_model_loader
from .data_loader import SlicedNiiDataset
from .eval import evaluate_model,evaluate_diffusion_model,evaluate_model_staple
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np

# Custom PyTorch Dataset
image_path = "../Datasets/Test/images"
label_path = "../Datasets/Test/labels"
tran_list = [transforms.Resize((256,256))]
transform_test = transforms.Compose(tran_list)
dataset = SlicedNiiDataset(image_path,label_path, transform=None)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,diffusion= custom_model_loader("../model/savedmodel105000.pt")
# Figure save dir
figure_dir = "../figures/eval_95000"

# Run Evaluate
evaluate_model_staple(model, data_loader, device, diffusion, num_samples=3,diffusion_step = 20,save_dir=figure_dir)
#evaluate_diffusion_model(model, data_loader, device, diffusion, num_ensemble=1, threshold=0.3)
#evaluate_model(model, data_loader, device)