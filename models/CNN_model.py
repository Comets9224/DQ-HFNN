import torch.nn as nn
"""
This file defines a classic neural network for the CIFAR-10 dataset.
Block class: This is a standard residual block (Residual Block), the core component of ResNet. It contains two convolutional layers, batch normalization (BatchNorm), and an optional shortcut connection.
myModel class:
Purpose: This is a complete ResNet-style classifier built upon residual blocks.
Structure: Using the make_layer function, it dynamically stacks residual blocks and pooling layers based on the configuration list cfg. Finally, a classifier fully connected layer outputs logits for 10 classes.
classical_layer class:
Purpose: This is the actual classic feature extractor used in MyNetwork_cifar10.
Difference from myModel: Its structure is nearly identical to myModel, with the only difference being the final classifier layer: nn.Linear(4 * 512, 256).
Its output is a 256-dimensional feature vector rather than logits for 10 classes. This is completely consistent with MnistModel's design philosophy: only extracting features without final classification.
"""
#cifar10
class Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True, stride=1):
        super(Block, self).__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False),
                nn.BatchNorm2d(outchannel),
            )
        else:
            self.shortcut = nn.Sequential()

        self.relu = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out += self.shortcut(x)
        out = self.relu(out)
        return out


class myModel(nn.Module):
    def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'], res=True):
        super(myModel, self).__init__()
        self.res = res       # Whether to use residual connections
        self.cfg = cfg       # Configuration list
        self.inchannel = 3   # Initial input channels
        self.futures = self.make_layer()
        # Build the fully connected layers and classifier after convolutional layers:
        self.classifier = nn.Sequential(nn.Dropout(0.4),         # Two fc layers perform slightly worse
                                        nn.Linear(4 * 512, 10), ) # fc layer, final output is 10 classes for Cifar10
    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class classical_layer(nn.Module):
    def __init__(self, cfg=[64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512,'M'], res=True):
        super(classical_layer, self).__init__()
        self.res = res       # Whether to use residual connections
        self.cfg = cfg       # Configuration list
        self.inchannel = 3   # Initial input channels
        self.futures = self.make_layer()
        # Build the fully connected layers after convolutional layers:
        self.classifier = nn.Sequential(nn.Dropout(0.4),         # Two fc layers perform slightly worse
                                        nn.Linear(4 * 512, 256), ) # fc layer, output 256-dim feature vector
    def make_layer(self):
        layers = []
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(Block(self.inchannel, v, self.res))
                self.inchannel = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.futures(x)
        # view(out.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out