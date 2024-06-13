import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from dataloader import CIFAR100Dataset, load_cifar100_data
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import warnings
from tqdm import tqdm
import os
warnings.filterwarnings("ignore")


# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


train_images, train_labels, test_images, test_labels = load_cifar100_data('../datasets/cifar-100-python')

# 创建Dataset对象
train_dataset = CIFAR100Dataset(train_images, train_labels, transform=transform)
test_dataset = CIFAR100Dataset(test_images, test_labels, transform=transform)

# 创建DataLoader对象
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载imagenet_val上自监督学习预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)
# 修改最后一层以适应 CIFAR-100 数据集（100 个类别）
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 100)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

# 仅训练最后一个线性分类层
for param in model.fc.parameters():
    param.requires_grad = True


# 准备优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 初始化 TensorBoard 记录器
writer = SummaryWriter('runs/sup_pretrain_cifar100_finetune')

# 训练模型
# 训练模型
num_epochs = 200
best_accuracy = 0.0


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print("Training...")
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    writer.add_scalar('training loss', epoch_loss, epoch)
    print(f"Loss: {epoch_loss:.4f}")

    print("Testing...")
    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    writer.add_scalar('testing accuracy', accuracy, epoch)
    print(f'Accuracy: {accuracy:.2f}%')
    print('----------------------------------------------')
    # 保存最好的模型权重
    if accuracy > best_accuracy:
        best_acc = accuracy
        best_epoch = epoch
        best_model_wts = model.state_dict()

# 保存最佳模型
best_model_path = 'weights/sup_pretrain_cifar100_finetune/'
os.makedirs(f'{best_model_path}', exist_ok=True)
torch.save(best_model_wts, f'{best_model_path}/best_model.pth')
print(f"Best model saved with accuracy: {best_acc}% at epoch {best_epoch}")

