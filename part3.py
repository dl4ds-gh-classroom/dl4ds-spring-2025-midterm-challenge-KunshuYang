import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import wandb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Configuration Section 
CONFIG = {
    "project": "ds542-part3-transfer-optimized-v3",
    "seed": 42,
    "batch_size": 128,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "data_dir": "./data",
    # Transfer learning related configurations
    "pretrained": True,
    "freeze_epochs": 3,           # Freeze phase: only train the fc layer
    "unfreeze_epochs": 147,       # Unfreeze phase: 147 epochs in total, making 150 epochs overall
    "base_lr": 0.01,
    "min_lr": 1e-5,
    "weight_decay": 5e-5,         # Reduced weight_decay to allow for increased capacity
    "label_smoothing": 0.1,
    "mixup_alpha": 0.4,
    # OOD test data directory (challenge data)
    "ood_dir": "./data/ood-test"
}

def get_model():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return model.to(CONFIG['device'])


# Data Augmentation Strategy (Including RandAugment)
def get_transforms():
    """
    Training data augmentation:
      - RandomResizedCrop, RandomHorizontalFlip, RandAugment (num_ops=1, magnitude=7), ColorJitter, Normalize, RandomErasing
    Testing data:
      - ToTensor and Normalize
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=1, magnitude=7),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform_train, transform_test


# MixUp Data Augmentation
def mixup_data(x, y, alpha=0.4):
    """Apply MixUp augmentation to the input"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Training Utility Functions
def train_epoch(model, loader, optimizer, criterion, epoch, phase):
    """Train or validate for one epoch"""
    if phase == 'train':
        model.train()
    else:
        model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(loader, desc=f"{phase} Epoch {epoch}", leave=False)
    for inputs, labels in progress_bar:
        inputs = inputs.to(CONFIG['device'], non_blocking=True)
        labels = labels.to(CONFIG['device'], non_blocking=True)
        
        if phase == 'train' and CONFIG['mixup_alpha'] > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, CONFIG['mixup_alpha'])
        
        with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            if phase == 'train' and CONFIG['mixup_alpha'] > 0:
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                loss = criterion(outputs, labels)
        
        if phase == 'train':
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_correct += outputs.argmax(1).eq(labels).sum().item()
        total_samples += batch_size
        
        progress_bar.set_postfix({
            "loss": total_loss / total_samples,
            "acc": total_correct / total_samples
        })
    return total_loss / total_samples, total_correct / total_samples

# OneCycleLR Scheduler Wrapper
def get_onecycle_scheduler(optimizer, num_steps, max_lr):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr, 
        total_steps=num_steps, 
        pct_start=0.15, 
        anneal_strategy='cos', 
        div_factor=25.0, 
        final_div_factor=1e4
    )

# Generate Submission File (using the eval_ood module)
def generate_submission(model, CONFIG):
    """
    Use the eval_ood module to predict on OOD test data and generate a submission file,
    conforming to the format of sample_submission.csv and saved as submission_part3.csv.
    """
    import eval_ood
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)
    submission_df = eval_ood.create_ood_df(all_predictions)
    submission_df.to_csv('submission_part3.csv', index=False)
    print("Submission file saved as submission_part3.csv")


# Main Training Pipeline
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    wandb.init(project=CONFIG['project'], config=CONFIG)
    
    transform_train, transform_test = get_transforms()
    trainset = torchvision.datasets.CIFAR100(root=CONFIG['data_dir'], train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=CONFIG['data_dir'], train=False, download=True, transform=transform_test)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])
    
    trainloader = DataLoader(trainset, batch_size=CONFIG['batch_size'], shuffle=True,
                             num_workers=CONFIG['num_workers'], pin_memory=True, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=CONFIG['batch_size'], shuffle=False,
                           num_workers=CONFIG['num_workers'], pin_memory=True)
    testloader = DataLoader(testset, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=CONFIG['num_workers'], pin_memory=True)
    
    model = get_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
    
    # ===== Phase 1: Freeze Phase, Train Only the fc Layer =====
    print("\n=== Phase 1: Training Last Layer Only ===")
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=CONFIG['base_lr'],
                            weight_decay=CONFIG['weight_decay'])
    scheduler_obj = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['freeze_epochs'])
    for epoch in range(CONFIG['freeze_epochs']):
        current_epoch = epoch + 1
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, current_epoch, 'train')
        val_loss, val_acc = train_epoch(model, valloader, optimizer, criterion, current_epoch, 'val')
        scheduler_obj.step()
        wandb.log({
            "epoch": current_epoch,
            "phase": "freeze",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        print(f"Epoch {current_epoch}/{CONFIG['freeze_epochs']} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # ===== Phase 2: Gradual Unfreezing =====
    print("\n=== Phase 2: Gradual Unfreezing ===")
    total_phase2_epochs = CONFIG['unfreeze_epochs']
    # Initially: Unfreeze layer4 (fc remains unfrozen)
    for name, param in model.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
    param_groups = [
        {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
        {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
    num_steps = len(trainloader) * total_phase2_epochs
    scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=0.025)
    
    for epoch in range(total_phase2_epochs):
        current_epoch = CONFIG['freeze_epochs'] + epoch + 1
        
        # At the middle of the unfreeze phase, unfreeze layer3
        if epoch == total_phase2_epochs // 2:
            print(">> Unfreezing layer3")
            for name, param in model.named_parameters():
                if 'layer3' in name:
                    param.requires_grad = True
            param_groups = [
                {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
                {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
                {"params": model.layer3.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.2},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
            num_steps = len(trainloader) * (total_phase2_epochs - epoch)
            scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=0.025)
        
        # At 3/4 of the unfreeze phase, unfreeze layer2
        if epoch == (3 * total_phase2_epochs) // 4:
            print(">> Unfreezing layer2")
            for name, param in model.named_parameters():
                if 'layer2' in name:
                    param.requires_grad = True
            param_groups = [
                {"params": model.fc.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 1.0},
                {"params": model.layer4.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.5},
                {"params": model.layer3.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.2},
                {"params": model.layer2.parameters(), "lr": CONFIG['base_lr'], "lr_mult": 0.1},
            ]
            optimizer = optim.AdamW(param_groups, weight_decay=CONFIG['weight_decay'])
            num_steps = len(trainloader) * (total_phase2_epochs - epoch)
            scheduler_oc = get_onecycle_scheduler(optimizer, num_steps, max_lr=0.025)
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion, current_epoch, 'train')
        val_loss, val_acc = train_epoch(model, valloader, optimizer, criterion, current_epoch, 'val')
        scheduler_oc.step()
        wandb.log({
            "epoch": current_epoch,
            "phase": "unfreeze",
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })
        print(f"Epoch {current_epoch}/{CONFIG['freeze_epochs']+total_phase2_epochs} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_acc > wandb.run.summary.get("best_val_acc", 0):
            wandb.run.summary["best_val_acc"] = val_acc
            torch.save(model.state_dict(), "best_model_part3.pth")
    
    print("\n=== Final Evaluation ===")
    model.load_state_dict(torch.load("best_model_part3.pth"))
    test_loss, test_acc = train_epoch(model, testloader, optimizer, criterion, 0, 'val')
    print(f"Test Accuracy: {test_acc:.2%}")
    wandb.log({"test_acc": test_acc})
    
    generate_submission(model, CONFIG)

if __name__ == '__main__':
    main()
