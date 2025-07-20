import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from tqdm import tqdm
import os
from model.resnet import ResNet34
from model.residual import ResidualBlock
from data.dataset import transform

def train(model, train_loader, optim, criterion, epoch, device):
    model.train()
    cur_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        inputs, labels = inputs.to(device), labels.to(device)

        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        cur_loss += loss.item()
        total += labels.size(0)
        predicted_classes = torch.argmax(outputs, dim=1)
        num_correct = (predicted_classes == labels).sum().item()
        correct += num_correct

    epoch_loss = cur_loss/len(train_loader)
    epoch_acc = 100*correct/total
    return epoch_loss, epoch_acc

def main():
    os.makedirs("checkpoints", exist_ok=True)

    # HYPERPARAMS
    epochs = 50
    batch_size = 128
    lr = 0.001
    num_classes = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34(ResidualBlock, [3,4,6,3], num_classes)
    criterion = optim.SGD(lr = lr)

    train_data = ds.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    val_data = ds.CIFAR10(root='./data', train=False, download=False, transform=transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)


    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optim, criterion, epoch, device)



