import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
import numpy as np
from torch.utils.data import random_split


# Model Definition - SimpleCNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # Conv Block 1: 3x32x32 -> 32x32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            # Conv Block 2: 32x16x16 -> 64x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), 
            
            # Conv Block 3: 64x8x8 -> 128x8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Training 
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    
    for i, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1),
            "acc": 100. * correct / total
        })
    
    return running_loss / len(trainloader), 100. * correct / total

def validate(model, valloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": running_loss / (i+1),
                "acc": 100. * correct / total
            })
    
    return running_loss/len(valloader), 100. * correct / total


# Main Function
def main():
    CONFIG = {
        "model": "SimpleCNN",
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 50,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "ds542-part1-cnn",
        "seed": 42,
    }
    torch.manual_seed(CONFIG["seed"])

    # Data Augmentation & Normalization
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], 
                           std=[0.2675, 0.2565, 0.2761]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                           std=[0.2675, 0.2565, 0.2761]),
    ])
    trainset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True,
        download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False,
        download=True, transform=transform_test
    )
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=CONFIG["num_workers"]
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"]
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=CONFIG["num_workers"]
    )

    model = SimpleCNN().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), 
                          lr=CONFIG["learning_rate"],
                          weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    import eval_cifar100
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

if __name__ == '__main__':
    main()