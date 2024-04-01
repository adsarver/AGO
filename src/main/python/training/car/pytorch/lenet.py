import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_channels, num_features):
        super(LeNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(num_channels, num_features, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(num_features, num_features * 2, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.fc1 = nn.Linear(num_features * 2 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(-1, num_features * 2 * 14 * 14)

        x = self.fc1(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)
        return x