import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
from alexnet import alexnet
import  matplotlib.pyplot as plt

weights_file = r'D:\研一下\深度学习\HW2\runs\alexnet_experiment_2\alexnet_cub200_best.pth'


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
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_dir = r'D:\研一下\深度学习\HW2\data\CUB_200_2011'
    test_dataset = CUB200Dataset(data_dir, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = alexnet(pretrained=False, num_classes=200,weights_file=weights_file).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

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
    print('Test Loss: {:.4f}'.format(running_loss / len(test_loader)))
    print('Test Accuracy: {:.4f}'.format(100 * correct / total))
if __name__ == '__main__':
    main()
