import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as ds
from tqdm import tqdm
import os
from model.resnet import ResNet34
from model.residual import ResidualBlock
from data.dataset import get_transform
import json

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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optim.step()

        cur_loss += loss.item()
        total += labels.size(0)
        predicted_classes = torch.argmax(outputs, dim=1)
        num_correct = (predicted_classes == labels).sum().item()
        correct += num_correct

    epoch_loss = cur_loss/len(train_loader)
    epoch_acc = 100*correct/total
    return epoch_loss, epoch_acc

def valid(model, val_loader, optim, criterion, epoch, device):
    model.eval()
    cur_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cur_loss += loss.item()
            total += labels.size(0)
            predicted_classes = torch.argmax(outputs, dim=1)
            num_correct = (predicted_classes == labels).sum().item()
            correct += num_correct

    val_loss = cur_loss / len(val_loader)
    val_acc = 100*correct/total
    return val_loss, val_acc



def main():
    os.makedirs("checkpoints", exist_ok=True)

    stats_path = "stats.json"
    checkpoint_path = "checkpoints/resnet34.pth"

    # HYPERPARAMS
    epochs = 200
    batch_size = 256
    lr = 0.1
    num_classes = 10
    
    train_transform, val_transform = get_transform()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34(ResidualBlock, [3,4,6,3], num_classes).to(device)

    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading weights from {checkpoint_path}: {e}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)

    train_data = ds.CIFAR10(root='./data', train=True, download=False, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    val_data = ds.CIFAR10(root='./data', train=False, download=False, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)

    best_val_acc = 0.0
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            best_val_acc = json.load(f)

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, device)
        val_loss, val_acc = valid(model, val_loader,  optim=optimizer, criterion=criterion, epoch=epoch, device=device)
        scheduler.step()

        print(f'Epoch {epoch+1}/{epochs}\nTraining loss: {train_loss}\nTraining accuracy: {train_acc}\nValidation loss: {val_loss}\nValidation accuracy: {val_acc}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/resnet34.pth')
            print(f'New best model with validation accuracy {best_val_acc} saved at run/checkpoints/resnet34.py')

            with open(stats_path, 'w') as f:
                json.dump(best_val_acc, f, indent=4)
            print(f'Updated {stats_path} with all-time best validation accuracy: {best_val_acc:.4f}')

if __name__ == "__main__":
    main()


