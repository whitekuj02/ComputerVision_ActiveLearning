
import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision.transforms import RandomVerticalFlip
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from Dataset.CUB2011 import CUB2011
import wandb

#Fix seeds for smooth comparisons
torch.manual_seed(42)

#adam  + randomVerticalFlip + 1e-4
### wandb setting ###
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CV_Active_1", 
    name = "SGD + 증강3개(호리즌탈, 로테, 가우시안)따로", #Rename every experiment

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1, 
    "architecture": "ResNet18",
    "dataset": "CUB2011",
    "augmentation": "RandomHorizontalFlip + andomRotation(30) + GaussianBlur(kernel_size=5) ",
    "epochs": "early_stop",
    "batch" : 32
    }
)

### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(DEVICE)
    
#data augmentation
transforms_train = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
])

aug_train1 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor()
])

aug_train2 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

aug_train3 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor()
])
# val, test does not augment
transforms_valtest = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
])

# augmentation As the number of batches increases, so does the number of batches, which is about 5.5 Gb / 24 Gb for 32
BATCH_SIZE = 32

train_set = CUB2011(mode='train',
                    transform=transforms_train)
val_set = CUB2011(mode='valid',
                    transform=transforms_valtest)
test_set = CUB2011(mode='test',
                    transform=transforms_valtest)
aug_set1 = CUB2011(mode='train', 
                    transform=aug_train1)
aug_set2 = CUB2011(mode='train', 
                    transform=aug_train2)
aug_set3 = CUB2011(mode='train', 
                    transform=aug_train3)

train_set = ConcatDataset([train_set, aug_set1, aug_set2, aug_set3])

print('Num of each dataset:', len(train_set), len(val_set), len(test_set))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print("Loaded dataloader")

### Model / Optimzier ###

# # The actual parameter adjustment is done here
EPOCH = 30 
lr = 0.1

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
### Transfer Learning ###
# replace the pre-trained resnet with 50 fc's to allow classification for 50 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 50)
model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=lr)

print("Created a learning model and optimizer")


def train(model, train_loader, optimizer, epoch):
    model.train()
    for i , (image,target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        optimizer.zero_grad()
        train_loss = F.cross_entropy(output, target).to(DEVICE)

        train_loss.backward()
        optimizer.step()

        if i%10==0:
            print(f'Train Epoch : {epoch} [{i}/{len(train_loader)}]\tLoss: {train_loss.item(): .6f} ')
            wandb.log({"train_loss": train_loss.item()})
    return train_loss

def evaluate(model, val_loader):
    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)

            eval_loss = F.cross_entropy(output, target, reduction='sum').item()

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss /= len(val_loader.dataset)
    eval_accuracy = 100 * correct / len(val_loader.dataset)
    return eval_loss, eval_accuracy

### Main ###

start = time.time()
best = 0
#early_stop = 0
#patience = 5

for epoch in range(EPOCH):
    # early_stop
    #if early_stop > patience:
    #    print(f"early_stop epochs : {epoch}")
    #    break
    
    # # train, test, and repeat for each epoch separately
    train_loss = train(model, train_loader, optimizer, epoch)
    val_loss, val_accuracy = evaluate(model, val_loader)

    wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss})

    # save bast model
    if val_accuracy > best : 
        best = val_accuracy
        torch.save(model.state_dict(), "./best_model.pth")
    #   early_stop = 0
    #else:
    #    early_stop += 1
        
    print(f'[{epoch} Validation Loss : {val_loss:.4f}, Accuracy: {val_accuracy:.4f}%]')

# Test result
test_loss, test_accuracy = evaluate(model, test_loader)
print(f'[FINAL] test Loss : {test_loss:.4f}, Accuracy: {test_accuracy:.4f}%')

wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss})

end = time.time()
elasped_time = end - start

print("Best Accuracy: ", best)
print(f"Elasped Time: {int(elasped_time/3600)}h, {int(elasped_time/60)}m, {int(elasped_time%60)}s")
print(f"time: {int(elasped_time/3600)}h, {int(elasped_time/60)}m, {int(elasped_time%60)}s")
wandb.finish()