import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets

def make_weights(images, nclasses, batch_size, num_workers=4):
    """
    Adapted from https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
    """

    device = torch.device("cuda")

    count = torch.zeros(nclasses).to(device)
    loader = DataLoader(images, batch_size=batch_size, num_workers=num_workers)

    for _, label in loader:
        label = label.to(device=device)
        idx, counts = label.unique(return_counts=True)
        count[idx] += counts

    N = count.sum()
    weight_per_class = N / count

    weight = torch.zeros(len(images)).to(device)

    for i, (img, label) in enumerate(loader):
        idx = torch.arange(0, img.shape[0]) + (i * batch_size)
        idx = idx.to(dtype=torch.long, device=device)
        weight[idx] = weight_per_class[label]

    return weight