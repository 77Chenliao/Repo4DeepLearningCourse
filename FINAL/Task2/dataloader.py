import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 自定义Dataset类来处理CIFAR-100数据
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

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据
train_images, train_labels, test_images, test_labels = load_cifar100_data('../datasets/cifar-100-python')

# 创建Dataset对象
train_dataset = CIFAR100Dataset(train_images, train_labels, transform=transform)
test_dataset = CIFAR100Dataset(test_images, test_labels, transform=transform)

# 创建DataLoader对象
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == '__main__':
    first_image = train_images[1]

    # 将图像数据从形状 (3072,) 转换为形状 (32, 32, 3)
    first_image = first_image.reshape(32, 32, 3)

    # 显示图像
    plt.imshow(first_image)
    plt.title(f"Label: {train_labels[0]}")
    plt.show()



    image, label = train_dataset[1]

    # 将图像转换回原始范围
    image = image / 2 + 0.5  # unnormalize
    image_np = image.numpy()
    
    # 显示图像
    plt.imshow(np.transpose(image_np, (1, 2, 0)))
    plt.title(f"Label: {label}")
    plt.show()
