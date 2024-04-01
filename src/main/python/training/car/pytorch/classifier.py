import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import flatten

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 512, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=5, padding=2)
        self.neurons = self.linear_input_neurons()

        # Fully connected layers
        # o1 = (image_width - kernel_width + 2*padding) / stride + 1
        # o2 = o1 / 2
        # o3 = (o2 - kernel_width + 2*padding) / stride + 1
        # o4 = o3 / 2
        # x = o4
        self.fc1 = nn.Linear(self.linear_input_neurons(), 128) # 16 * 5 * 5 : conv2 output channels * x * x
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print(x.size())
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))
        print(x.size())
        x = flatten(x, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return  self.fc3(x)
    
    # here we apply convolution operations before linear layer, and it returns the 4-dimensional size tensor. 
    def size_after_relu(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x.float())))

        return x.size()


    # after obtaining the size in above method, we call it and multiply all elements of the returned size.
    def linear_input_neurons(self):
        size = self.size_after_relu(torch.rand(6, 3, 512, 512)) # image size: 64x32
        m = 1
        for i in size:
            m *= i

        return int(m)