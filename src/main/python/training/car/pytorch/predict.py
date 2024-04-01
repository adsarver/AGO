from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor

trainData = KMNIST(root="data", train=True, download=True,
	transform=ToTensor())