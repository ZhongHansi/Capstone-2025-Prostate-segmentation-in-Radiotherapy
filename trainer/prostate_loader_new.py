import torch
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from PIL import Image


class ProstateDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform=None, image_size=256, slice_mode=True):
        """
        专门针对前列腺数据的加载器
        数据格式: [H, W, D, C] = [320, 320, 15, 2]

        Args:
            directory (str): 训练数据目录路径
            transform (callable, optional): 预处理数据的转换函数
            image_size (int): 输出图像的大小
            slice_mode (bool): 是否以切片模式加载数据
        """
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.slice_mode = slice_mode
        self.image_size = image_size  # 添加image_size属性

        # 获取 images/ 和 labels/ 目录
        self.image_dir = os.path.join(self.directory, "images")
        self.label_dir = os.path.join(self.directory, "labels")

        # 获取所有的 image 文件
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".nii.gz")])

        self.database = []
        for image_file in self.image_files:
            image_path = os.path.join(self.image_dir, image_file)
            label_path = os.path.join(self.label_dir, image_file)

            if os.path.exists(label_path):  # 确保 label 存在
                self.database.append((image_path, label_path))

        print(f"✅ Loaded {len(self.database)} prostate samples from {self.directory}")

        # 检查第一个样本确定切片数
        if len(self.database) > 0:
            sample_img = nib.load(self.database[0][0])
            sample_data = sample_img.get_fdata()
            self.slices_per_volume = sample_data.shape[2]  # 深度维度的大小
            print(f"Each volume has {self.slices_per_volume} slices")

    def __len__(self):
        if self.slice_mode:
            return len(self.database) * self.slices_per_volume
        else:
            return len(self.database)

    def __getitem__(self, idx):
        if self.slice_mode:
            # 切片模式 - 返回单个2D切片
            volume_idx = idx // self.slices_per_volume
            slice_idx = idx % self.slices_per_volume

            # 确保索引在范围内
            volume_idx = min(volume_idx, len(self.database) - 1)

            image_path, label_path = self.database[volume_idx]

            # 加载数据
            image_nib = nib.load(image_path)
            label_nib = nib.load(label_path)

            # 获取原始方向和仿射矩阵，确保图像和标签使用相同的空间参考
            image_affine = image_nib.affine

            image_data = image_nib.get_fdata()  # [H, W, D, C]
            label_data = label_nib.get_fdata()  # [H, W, D, C] 或 [H, W, D]

            # 确保切片索引不超出实际切片数
            actual_slices = image_data.shape[2]
            slice_idx = min(slice_idx, actual_slices - 1)

            # 提取指定切片 - 不进行任何重新排列或翻转
            if len(image_data.shape) == 4:  # [H, W, D, C]
                image_slice = image_data[:, :, slice_idx, :]  # 使用双通道 [H, W, C]
            else:  # [H, W, D]
                image_slice = image_data[:, :, slice_idx]  # [H, W]

            if len(label_data.shape) == 4:  # [H, W, D, C]
                label_slice = label_data[:, :, slice_idx, 0]  # [H, W]
            else:  # [H, W, D]
                label_slice = label_data[:, :, slice_idx]  # [H, W]

            # 保持原始方向，不要翻转或旋转
            image_np = image_slice.astype(np.float32)
            label_np = (label_slice > 0.5).astype(np.float32)  # 二值化标签

            # 归一化图像值到[0,1]范围
            for c in range(image_np.shape[-1]):
                ch = image_np[:, :, c]
                if ch.max() > 0:
                    image_np[:, :, c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

            # 直接调整大小，不转换为PIL图像（避免潜在的翻转）
            image_np = np.transpose(image_np, (2, 0, 1))  # [C, H, W]
            image_tensor = torch.from_numpy(image_np).float()
            label_tensor = torch.from_numpy(label_np).float().unsqueeze(0)  # [1, H, W]

            # 使用F.interpolate调整大小，确保保持原始方向
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),  # [1, 1, H, W]
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [1, image_size, image_size]

            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0),  # [1, 1, H, W]
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(0)  # [1, image_size, image_size]

            # 如果需要应用其他变换
            if self.transform:
                image_tensor = self.transform(image_tensor)
                label_tensor = self.transform(label_tensor)

            return image_tensor, label_tensor, f"{image_path}_slice{slice_idx}"
        else:
            # 非切片模式代码...
            image_path, label_path = self.database[idx]

            # 加载数据
            image_nib = nib.load(image_path)
            label_nib = nib.load(label_path)

            image_data = image_nib.get_fdata()  # [H, W, D, C]
            label_data = label_nib.get_fdata()  # [H, W, D, C] 或 [H, W, D]

            # 使用最大值投影创建一个2D视图，保持原始方向
            if len(image_data.shape) == 4:  # [H, W, D, C]
                image_projection = np.max(image_data[:, :, :, 0], axis=2)  # [H, W]
            else:
                image_projection = np.max(image_data, axis=2)  # [H, W]

            if len(label_data.shape) == 4:  # [H, W, D, C]
                label_projection = np.max(label_data[:, :, :, 0], axis=2)  # [H, W]
            else:
                label_projection = np.max(label_data, axis=2)  # [H, W]

            # 转换为PyTorch张量
            image_tensor = torch.from_numpy(image_projection).float().unsqueeze(0)  # [1, H, W]
            label_tensor = torch.from_numpy(label_projection).float().unsqueeze(0)  # [1, H, W]

            # 确保标签是二值的
            label_tensor = (label_tensor > 0.5).float()

            # 调整大小为需要的尺寸
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),  # [1, 1, H, W]
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # [1, image_size, image_size]

            label_tensor = F.interpolate(
                label_tensor.unsqueeze(0),  # [1, 1, H, W]
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(0)  # [1, image_size, image_size]

            # 归一化
            if image_tensor.max() > 0:
                image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())

            # 应用变换（如果有）
            if self.transform:
                image_tensor = self.transform(image_tensor)
                label_tensor = self.transform(label_tensor)

            return image_tensor, label_tensor, f"{image_path}_idx{idx}"