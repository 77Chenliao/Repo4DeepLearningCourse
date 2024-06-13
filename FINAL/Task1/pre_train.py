import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import random_split
import warnings
import os
warnings.filterwarnings("ignore")

from dataloader import FashionMNISTDataset, RotatedFashionMNISTDataset
from dataloader import ImageNetDataset, RotatedImageNetDataset

dataset = 'imagenet'

if dataset == "imagenet":
    imagenet_val_dir = '../datasets/imagenet_val'
    train_dataset = ImageNetDataset(imagenet_val_dir)
    rotated_train_dataset = RotatedImageNetDataset(train_dataset)
    train_size = int(0.8 * len(rotated_train_dataset))
    test_size = len(rotated_train_dataset) - train_size
    rotated_train_dataset, rotated_test_dataset = random_split(rotated_train_dataset, [train_size, test_size])
    rotated_train_loader = torch.utils.data.DataLoader(rotated_train_dataset, batch_size=64, shuffle=True)
    rotated_test_loader = torch.utils.data.DataLoader(rotated_test_dataset, batch_size=64, shuffle=False)
else:
    train_dataset = FashionMNISTDataset('../datasets/FashionMNIST/train-images-idx3-ubyte',
                                        '../datasets/FashionMNIST/train-labels-idx1-ubyte')
    rotated_train_dataset = RotatedFashionMNISTDataset(train_dataset)
    rotated_train_loader = torch.utils.data.DataLoader(rotated_train_dataset, batch_size=64, shuffle=True)

    test_dataset = FashionMNISTDataset('../datasets/FashionMNIST/t10k-images-idx3-ubyte',
                                       '../datasets/FashionMNIST/t10k-labels-idx1-ubyte')
    rotated_test_dataset = RotatedFashionMNISTDataset(test_dataset)
    rotated_test_loader = torch.utils.data.DataLoader(rotated_test_dataset, batch_size=64, shuffle=False)

# prepare model
model = models.resnet18(pretrained=False)

# 修改最后一层，以适应旋转预测任务
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)  # 4 个类别分别对应 0 度、90 度、180 度和 270 度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# prepare optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 50

writer = SummaryWriter(f'runs/{dataset}')

# train model
train_loss_list = []
best_acc = 0
best_epoch = 0
best_model_wts = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(rotated_train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        running_loss += loss.item() * images.size(0)

    writer.add_scalar('training loss', running_loss / len(rotated_train_loader.dataset), epoch)

    epoch_loss = running_loss / len(rotated_train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}]: Loss: {epoch_loss:.4f}")

    # 测试集上测试并保存最佳模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in rotated_test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy on test images: {acc}%")
    writer.add_scalar('testing accuracy', acc, epoch)

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_model_wts = model.state_dict()

# 保存最佳模型
os.makedirs(f'./weights/{dataset}_pretrain', exist_ok=True)
torch.save(best_model_wts, f'./weights/{dataset}_pretrain/best_model.pth')
print(f"Best model saved with accuracy: {best_acc}% at epoch {best_epoch}")
