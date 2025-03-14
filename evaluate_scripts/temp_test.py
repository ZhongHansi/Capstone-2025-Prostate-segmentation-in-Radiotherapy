from .eval import evaluate_model
import torch
import torchvision.transforms as transforms
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from .model_loader import custom_model_loader

# Load Dataset
data_path = "../Datasets/Test/images"
tran_list = [transforms.Resize((256,256)), transforms.ToTensor()]
transform_test = transforms.Compose(tran_list)

nii_files = [f for f in os.listdir(data_path) if f.endswith(".nii") or f.endswith(".nii.gz")]

image_list = []
for file in nii_files:
    file_path = os.path.join(data_path, file)
    nii_image = nib.load(file_path)  # read NIfTI files
    image_data = nii_image.get_fdata()  # trans to NumPy array
    image_list.append(image_data)

    print(f"Loaded: {file}, Shape: {image_data.shape}, Dtype: {image_data.dtype}")

# Load model and do evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = custom_model_loader("../model/savedmodel105000.pt")
#print(model)
dummy_input = torch.randn(1, 2, 256, 256).to(device)  # (batch_size=1, in_channels=2, H=256, W=256)
timesteps = torch.tensor([50],device=device)
output = model(dummy_input,timesteps)

# print output shape
print(type(output)) 
print(len(output))   
print(output[0].shape) 

pred = output[0].squeeze(0)
print("Unique values in pred:", torch.unique(pred[0]))
print(pred.shape)
#plt.subplot(1, 2, 1)
#plt.imshow(pred[0].cpu().detach().numpy(), cmap="gray")
#plt.title("Output Channel 0")

#plt.subplot(1, 2, 2)
#plt.imshow(pred[1].cpu().detach().numpy(), cmap="gray")
#plt.title("Output Channel 1")

#plt.show()
def show_image_sample():
    sample_idx = 0  
    image_data = image_list[sample_idx]
    slice_idx = image_data.shape[2] // 2 
    slice_img = image_data[:, :, slice_idx]
    slice_img = slice_img[:,:,1]
    print("Unique values in target:", torch.unique(torch.tensor(slice_img)))
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Middle Slice of {nii_files[sample_idx]}")
    plt.axis("off")
    plt.show()

show_image_sample()