import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000,weights_file=None):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        if weights_file:
            state_dict = torch.load(weights_file)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, model_root=None, num_classes=200,weights_file=None):
    if pretrained and not weights_file:  # 替换掉最后一层
        model = AlexNet(num_classes=1000)
        state_dict = model_zoo.load_url(model_urls['alexnet'], model_root)
        # Load the pretrained weights, except the final layer
        model.load_state_dict(state_dict, strict=False)
        # Reset the final layer for 200 classes
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif pretrained and weights_file:
        # 抛出异常
        raise ValueError("Cannot have both pretrained and weights_file")
    else:
        model = AlexNet(num_classes=num_classes,weights_file=weights_file)
    return model