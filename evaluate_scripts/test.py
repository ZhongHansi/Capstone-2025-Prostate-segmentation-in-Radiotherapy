from .model_loader import custom_model_loader
from .data_loader import SlicedNiiDataset
from .eval import evaluate_model
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
dataset = SlicedNiiDataset(image_path,label_path)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_model_loader("../model/savedmodel105000.pt")


# Run Evaluate
evaluate_model(model, data_loader, device)
