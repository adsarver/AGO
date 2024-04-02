import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import flatten

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 256, kernel_size=5)

        # Fully connected layers
        # o1 = (image_width - kernel_width + 2*padding) / stride + 1
        # o2 = o1 / 2
        # o3 = (o2 - kernel_width + 2*padding) / stride + 1
        # o4 = o3 / 2
        # x = o4
        self.fc1 = nn.Linear(256*125*125, 128) # 16 * 5 * 5 : conv2 output channels * x * x
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 2801)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        print(x.shape)
        x = x.view(-1, 256*125*125)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return  self.fc3(x)