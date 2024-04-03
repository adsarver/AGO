import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import flatten

class Classifier(nn.Module):
    classes = 0
    imgdims = (512, 512)
    kernel_width = 5
    padding = 0
    stride = 1
    convout = 128
    def __init__(self, classes, imgdims, kernel_width=5, padding=0, stride=1):
        super().__init__()
        self.classes = classes
        self.imgdims = imgdims
        self.kernel_width = kernel_width
        self.padding = padding
        self.stride = stride
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=kernel_width)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, self.convout, kernel_size=kernel_width)

        # Fully connected layers
        # o1 = (image_width - kernel_width + 2*padding) / stride + 1
        # o2 = o1 / 2
        # o3 = (o2 - kernel_width + 2*padding) / stride + 1
        # o4 = o3 / 2
        # x = o4
        o_calc = self.o_calc()
        self.fc1 = nn.Linear(self.convout*o_calc*o_calc, 128) # 16 * 5 * 5 : conv2 output channels * x * x
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        o_calc = self.o_calc()
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        x = x.view(-1, self.convout*o_calc*o_calc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # must return shape [classes, 1, h, w]
        return x
    
    def o_calc(self):
        o1 = (self.imgdims[0] - self.kernel_width + 2*self.padding) / self.stride + 1
        o2 = o1 / 2
        o3 = (o2 - self.kernel_width + 2*self.padding) / self.stride + 1
        o4 = o3 / 2
        return int(o4)