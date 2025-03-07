from .eval import evaluate_model
import torch
import torchvision.transforms as transforms
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import argparse
from MedsegDiff_V1_Model.UNet import UNetModel_newpreview
from MedsegDiff_V1_Model.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
# Load Dataset
data_path = "../Datasets/Dataset001_Prostate158/test"
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
# load model
image_size = 256
attention_resolutions = "16"
attention_ds = []
for res in attention_resolutions.split(","):
    attention_ds.append(image_size // int(res))
model = UNetModel_newpreview(image_size=256,
                            in_channels=2,
                            model_channels=128,
                            out_channels=2,
                            num_res_blocks=2,
                            attention_resolutions=tuple(attention_ds),)
model = model.to(device)
model_path = "../model/savedmodel105000.pt"
model.load_part_state_dict(torch.load(model_path))

model.eval()  
print("Load Model Success")


def show_image_sample():
    sample_idx = 0  
    image_data = image_list[sample_idx]
    slice_idx = image_data.shape[2] // 2 
    slice_img = image_data[:, :, slice_idx]
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Middle Slice of {nii_files[sample_idx]}")
    plt.axis("off")
    plt.show()

