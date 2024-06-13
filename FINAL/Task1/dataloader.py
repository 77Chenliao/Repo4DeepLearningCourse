import os
import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import pickle
import numpy as np
import os

# 设置seed
SEED = 2024
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# prepare dataset and dataloader
class FashionMNISTDataset(Dataset,):
    def __init__(self, image_path, label_path):
        self.images = self.read_images(image_path)
        self.labels = self.read_labels(label_path)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def read_images(self, path):
        with open(path, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def read_labels(self, path):
        with open(path, 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image, mode='L')
        if self.transform:
            image = self.transform(image)
        return image, label

class RotatedFashionMNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        angle = np.random.choice(self.angles)
        rotated_image = transforms.functional.rotate(image, int(angle))
        angle_idx = self.angles.index(angle)
        return rotated_image, angle_idx


class ImageNetDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整图像大小为 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 使用 ImageNet 数据集的均值和标准差进行标准化
])
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


class RotatedImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]
        angle = random.choice(self.angles)
        rotated_image = transforms.functional.rotate(image, angle)
        angle_idx = self.angles.index(angle)
        return rotated_image, angle_idx


class CIFAR100Dataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # 先转换为形状 (H, W, C)
        image = image.reshape(32, 32, 3).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, label


# 解包数据文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# 加载CIFAR-100数据
def load_cifar100_data(path):
    train_data = unpickle(f"{path}/train")
    test_data = unpickle(f"{path}/test")

    train_images = train_data[b'data']
    train_labels = train_data[b'fine_labels']

    test_images = test_data[b'data']
    test_labels = test_data[b'fine_labels']

    return train_images, train_labels, test_images, test_labels


