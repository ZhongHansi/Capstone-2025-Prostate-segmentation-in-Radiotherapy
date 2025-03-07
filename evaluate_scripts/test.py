from MedsegDiff_V1_Model.model import MedSegDiffV1
from .eval import evaluate_model
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.utils.data as data
import nibabel as nib
import numpy as np

# Custom PyTorch Dataset
class NIfTIDataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        
        self.data_dir = data_dir
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        # read NIfTI file
        nii_image = nib.load(file_path)
        image_data = nii_image.get_fdata()

        # process on 4D
        if len(image_data.shape) == 4:
            image_data = image_data[..., 0]  # (H, W, D)

        # normalize
        image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))

        # transfer to PyTorch Tensor as (1, D, H, W)
        image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, file_name


# Create DataLoader

data_dir = "../Datasets/Dataset001_Prostate158/test"
dataset = NIfTIDataset(data_dir)
data_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)


# Load model and do evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model
model_path = "../model/savedmodel105000.pt"
model = torch.load(model_path)
print(model)
model = model.to(device)
model.eval()  
print("Load Model Success")

with torch.no_grad():
    for images, file_names in data_loader:
        images = images.to(device) 
        outputs = model(images)  # predict

        print(f"Processed: {file_names[0]}, Output Shape: {outputs.shape}")  # print prediction

# Run Evaluate
evaluate_model(model, data, device)
