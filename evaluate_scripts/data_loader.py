import torch
import torchvision.transforms as transforms
import os
import nibabel as nib
from torch.utils.data import Dataset,DataLoader
import numpy as np
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




class SlicedNiiDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.img_nii_files = [f for f in os.listdir(img_path) if f.endswith(".nii") or f.endswith(".nii.gz")]
        self.label_nii_files = [f for f in os.listdir(label_path) if f.endswith(".nii") or f.endswith(".nii.gz")]
        self.image_slices = []
        self.label_slices = []
        for file in self.img_nii_files:
            file_path = os.path.join(img_path, file)
            nii_image = nib.load(file_path)
            image_data = nii_image.get_fdata()  # (H, W, D, C)

            image_data = np.transpose(image_data, (2, 3, 0, 1))
            slice_idx = image_data.shape[0] // 2 
            self.image_slices.append(image_data[slice_idx])

        for file in self.label_nii_files:
            file_path = os.path.join(label_path, file)
            nii_label = nib.load(file_path)
            label_data = nii_label.get_fdata()  # (H, W, D, C)

            
            label_data = np.transpose(label_data, (2, 0, 1))
            slice_idx = label_data.shape[0] // 2 
            self.label_slices.append(label_data[slice_idx])

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        label = self.label_slices[idx]

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        if self.transform:
            image = self.transform(image)  
            label = self.transform(label)
           
        return image,label

img_path = "../Datasets/Test/images"
label_path = "../Datasets/Test/labels"
dataset = SlicedNiiDataset(img_path,label_path)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
for images, labels in data_loader:
    print(f"Batch Image Shape: {images.shape}")  # (batch_size, 2, 256, 256)
    print(f"Batch Label Shape: {labels.shape}")
    break
