import torchvision.datasets as ds
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch

def mean_std(dataset, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    pixels = 0

    for imgs, _ in tqdm(loader):
        channel_sum += imgs.sum(dim=[0,2,3])
        channel_sum_sq += (imgs**2).sum(dim=[0,2,3])
        pixels += imgs.shape[0] * imgs.shape[2] * imgs.shape[3]

    # image dataset stats formula
    mean = channel_sum / pixels
    std = torch.sqrt((channel_sum_sq / pixels) - (mean ** 2))
    return mean, std

def get_transform():
    temp_transform = transforms.ToTensor()

    dataset = ds.CIFAR10(root='./data', train=True, download=True, transform=temp_transform)

    mean, std = mean_std(dataset, batch_size=128)

    train_transform = transforms.Compose([
        transforms.RandomErasing(scale=(0.02, 0.33), ratio=(0.33, 3.3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    return train_transform, val_transform

if __name__ == "__main__":
    transform = get_transform()