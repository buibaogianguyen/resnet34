import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

def mean_std(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    channel_sum = np.zeros(3)
    channel_sum_sq = np.zeros(3)
    pixels = 0

    for imgs, _ in tqdm(loader):
        channel_sum += imgs.sum(dim=[0,2,3])
        channel_sum_sq += (imgs**2).sum(dim=[0,2,3])
        pixels += imgs.shape[1] * imgs.shape[2]

    # image dataset stats formula
    mean = channel_sum / pixels
    std = np.sqrt((channel_sum_sq / pixels) - (mean ** 2))
    return mean, std

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),    
])

dataset = ds.CIFAR10(root='./data', train=True, download=True, transform=transform)

mean, std = mean_std(dataset, batch_size=128)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])