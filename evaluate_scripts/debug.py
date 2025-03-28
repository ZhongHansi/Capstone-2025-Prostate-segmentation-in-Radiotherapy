from .eval import evaluate_model, target_plot
from .data_loader import SlicedNiiDataset
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from .model_loader import custom_model_loader

# Load Dataset
data_path = "../Datasets/Test/labels"
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

# Custom PyTorch Dataset
image_path = "../Datasets/Test/images"
label_path = "../Datasets/Test/labels"
tran_list = [transforms.Resize((256,256)),transforms.Normalize([0.5, 0.5], [0.5, 0.5])]
transform_test = transforms.Compose(tran_list)
dataset = SlicedNiiDataset(image_path,label_path, transform=None)
batch_size = 1
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# Load model and do evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model,diffusion= custom_model_loader("../model/3.27/savedmodel010000.pt")
#print(model)
dummy_input = torch.randn(1, 2, 320, 320).to(device)  # (batch_size=1, in_channels=2, H=256, W=256)
timesteps = torch.tensor([0],device=device)
#output = model(dummy_input,timesteps)

# print output shape
#print(type(output)) 
#print(len(output))   
#print(output[0].shape) 

#pred = output[0].squeeze(0)
#print("Unique values in pred:", torch.unique(pred[0]))
#print(pred.shape)
#print(pred.type)
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
    #slice_img = slice_img[:,:,1]
    print("Unique values in target:", torch.unique(torch.tensor(slice_img)))
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Middle Slice of {nii_files[sample_idx]}")
    plt.axis("off")
    plt.show()

#show_image_sample()
import matplotlib.pyplot as plt

model.eval()

with torch.no_grad():
    img, label = next(iter(data_loader))
    img, label = img.to(device), label.to(device)
    model_kwargs={}
    sample, *_ = diffusion.p_sample_loop_known(
        model,
        img.shape,
        img,      # conditioning 
        step = 100,               
        clip_denoised=True,
        model_kwargs=model_kwargs,
    )

    pred = torch.sigmoid(sample)
    print(pred.shape)
    pred_bin = (pred > 0.3).float()

    # squeeze
    img_vis = img.squeeze().cpu()
    pred_vis = pred_bin.squeeze().cpu()
    label_vis = label.squeeze().cpu()

    # plot
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(img_vis[0], cmap='gray'); plt.title('Input Image')
    plt.subplot(1,3,2); plt.imshow(pred_vis, cmap='gray'); plt.title('Predicted Mask')
    plt.subplot(1,3,3); plt.imshow(label_vis, cmap='gray'); plt.title('Ground Truth')
    plt.tight_layout()
    plt.show()
