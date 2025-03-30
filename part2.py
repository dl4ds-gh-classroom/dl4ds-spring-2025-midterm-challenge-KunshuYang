# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from tqdm.auto import tqdm
import wandb
import os

# ---------------------------
# Training Function
# ---------------------------
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)
    for i, (inputs, labels) in enumerate(progress_bar):
        # Move data to device
        inputs, labels = inputs.to(CONFIG['device']), labels.to(CONFIG['device'])
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            "loss": running_loss / (i + 1),
            "acc": 100. * correct / total
        })

    return running_loss / len(trainloader), 100. * correct / total

# ---------------------------
# Validation Function
# ---------------------------
def validate(model, valloader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)
        for i, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                "loss": running_loss / (i+1),
                "acc": 100. * correct / total
            })

    return running_loss / len(valloader), 100. * correct / total

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Configuration dictionary
    CONFIG = {
        "model": "ResNet18",
        "batch_size": 128,
        "learning_rate": 0.001,
        "epochs": 50,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",
        "ood_dir": "./data/ood-test",
        "wandb_project": "ds542-part2-resnet",
        "seed": 42,
    }
    torch.manual_seed(CONFIG["seed"])

    # Data augmentation and normalization
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

    # Load CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=True,
        download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root=CONFIG["data_dir"], train=False,
        download=True, transform=transform_test
    )

    # Split into train/val sets (80/20)
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    # Create data loaders
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

    # ---------------------------
    # Initialize ResNet18 model
    # ---------------------------
    model = torchvision.models.resnet18(pretrained=False)  # Set pretrained=True for transfer learning
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)  # Modify final layer for CIFAR-100 (100 classes)
    model = model.to(CONFIG["device"])

    # Loss function, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    # Initialize Weights & Biases for experiment tracking
    wandb.init(project=CONFIG["wandb_project"], config=CONFIG)
    wandb.watch(model)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(CONFIG["epochs"]):
        # Train and validate
        train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
        val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
        scheduler.step()

        # Log metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            wandb.save("best_model.pth")

    wandb.finish()

    # Evaluate model on test set
    import eval_cifar100
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

if __name__ == '__main__':
    main()