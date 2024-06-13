import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataloader import train_loader, test_loader
from torchvision.models import resnet50
from transformer_model import VisionTransformer
from CutMix import  cutmix_data
import  os


# 选择模型
model_name = 'resnet50'  # 'resnet50' or 'vit'

if model_name == 'resnet50':
    model = resnet50(weights=None, num_classes=100)
elif model_name == 'vit':
    model = VisionTransformer()

# 将模型移动到GPU（如果可用）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 定义损失函数、优化器和学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 创建SummaryWriter
writer = SummaryWriter(log_dir=rf'logs/{model_name}')

# 训练和评估模型
num_epochs = 100
best_model_wts = model.state_dict()
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # 训练阶段
    model.train()
    running_loss = 0.0
    running_corrects = 0
    alpha = 0.5

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)


        inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha, device)

        optimizer.zero_grad()

        # 前向传播
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        running_corrects += lam * torch.sum(preds == targets_a.data) + (1 - lam) * torch.sum(preds == targets_b.data)

    exp_lr_scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # 记录训练损失和准确率
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Accuracy', epoch_acc, epoch)

    # 每10个epoch进行一次验证
    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(test_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(test_loader.dataset)

        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')

        # 记录验证损失和准确率
        writer.add_scalar('Val/Loss', val_epoch_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_epoch_acc, epoch)

        # 深拷贝模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = model.state_dict()

print(f'Best val Acc: {best_acc:.4f}')

# 加载最佳模型权重
model.load_state_dict(best_model_wts)
weights_path = f'weights/{model_name}'
os.makedirs(weights_path, exist_ok=True)
torch.save(model.state_dict(), f"{weights_path}/best_model.pth")
# 关闭SummaryWriter
writer.close()
