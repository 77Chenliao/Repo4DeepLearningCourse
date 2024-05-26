import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from alexnet import alexnet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json

PRETRAINED = False


class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 读取图像路径和标签
        image_path = os.path.join(root_dir, 'images.txt')
        label_path = os.path.join(root_dir, 'image_class_labels.txt')
        split_path = os.path.join(root_dir, 'train_test_split.txt')

        image_dict = {line.split()[0]: line.split()[1] for line in open(image_path, 'r')}
        label_dict = {line.split()[0]: int(line.split()[1])-1 for line in open(label_path, 'r')}
        split_dict = {line.split()[0]: int(line.split()[1]) for line in open(split_path, 'r')}

        for img_id in image_dict:
            if (split_dict[img_id] == 1 and split == 'train') or (split_dict[img_id] == 0 and split == 'test'):
                self.images.append(os.path.join(root_dir, 'images', image_dict[img_id]))
                self.labels.append(label_dict[img_id])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def main():
    writer = SummaryWriter('runs/alexnet_experiment_1')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = r'D:\研一下\深度学习\HW2\data\CUB_200_2011'
    train_dataset = CUB200Dataset(data_dir, split='train', transform=transform)
    test_dataset = CUB200Dataset(data_dir, split='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = alexnet(pretrained=PRETRAINED, num_classes=200).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    epochs = 50

    train_losses = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    best_model_path = 'alexnet_cub200_best.pth'

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        writer.add_scalar('training loss', train_losses[-1], epoch)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(running_loss / len(test_loader))
        test_accuracies.append(correct / total)
        writer.add_scalar('test loss', test_losses[-1], epoch)
        writer.add_scalar('test accuracy', test_accuracies[-1], epoch)
        print(
            f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.2f}')

        # 保存最佳模型
        if test_accuracies[-1] > best_accuracy:
            best_accuracy = test_accuracies[-1]
            torch.save(model.state_dict(), best_model_path)
    writer.close()

    file_name = 'pretrained_losses' if PRETRAINED else 'random_losses'
    with open(f'{file_name}.json', 'w') as f:
        json.dump({'train_losses': train_losses, 'test_losses': test_losses, 'test_accuracies': test_accuracies}, f,
                  indent=4)

    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(test_losses, label='Test Loss')
    # plt.title('Loss over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(test_accuracies)
    # plt.title('Accuracy over epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy (%)')
    # plt.show()


if __name__ == '__main__':
    main()
