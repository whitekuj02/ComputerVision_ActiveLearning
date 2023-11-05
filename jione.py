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
import cv2

#원활한 비교를 위해 시드 고정
torch.manual_seed(42)

#adam  + randomVerticalFlip + 1e-4
### wandb setting ###
# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="CV_Active_1", # 프로젝트 명은 그대로
    name = "(SGD + 증강4개(호리즌탈, 로테30, 가우시안, 캐니)따로 (jione))(최고) + 배치 32 + 얼리스탑 + softloss", # 매 실험마다 이름 바꾸어주기

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.1, # 이건 조금 줄여서 세심하게 탐사 해봐야 할 듯
    "architecture": "ResNet18",
    "dataset": "CUB2011",
    "augmentation": "RandomHorizontalFlip + andomRotation(30) + GaussianBlur(kernel_size=11) + 캐니",
    "epochs": "early stop",
    "batch" : 32
    }
)

### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(DEVICE)

# 이미지에 Canny 에지 감지 적용하는 함수
def apply_canny_edge_detection(image, threshold1=100, threshold2=200):
    # 이미지를 NumPy 배열로 변환
    image_np = np.array(image)
    
    # 이미지를 그레이 스케일로 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # Canny 에지 감지 적용
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # 에지 이미지를 3채널로 확장
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    
    # PIL 이미지로 변환
    edges_image = Image.fromarray(edges_rgb)
    
    return edges_image

# Canny 에지 감지를 적용할 augment transform 함수
class CannyEdgeDetection(object):
    def __init__(self, threshold1=100, threshold2=200):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def __call__(self, image):
        return apply_canny_edge_detection(image, self.threshold1, self.threshold2)
    
    
# augmentation 하면 될 듯
transforms_train = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
])

aug_train1 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
])

aug_train2 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
])

aug_train3 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.GaussianBlur(kernel_size=11),
    transforms.ToTensor(),
])

# torchvision.transforms.Compose에 포함시킬 수 있도록 CannyEdgeDetection을 사용한 transform 정의
aug_train4 = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    CannyEdgeDetection(threshold1=100, threshold2=200),
    transforms.ToTensor(),
])


# val, test는 augmentation 하지 않음
transforms_valtest = transforms.Compose([
    transforms.Resize((448, 448)), 
    transforms.ToTensor(),
])

# augmentation 늘어날 수록 batch 수도 늘려서 32 기준 5.5 Gb / 24Gb 정도
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
aug_set4 = CUB2011(mode='train', 
                    transform=aug_train4)

train_set = ConcatDataset([train_set, aug_set1, aug_set2, aug_set3, aug_set4])

print('Num of each dataset:', len(train_set), len(val_set), len(test_set))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print("Loaded dataloader")

### Model / Optimzier ###

# 실제 parameter 조정은 여기서
EPOCH = 30 
lr = 0.1

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
### Transfer Learning ###
# pre-train된 resnet 뒤에 fc 50개 짜리로 바꾸어서 50개의 class에 대한 classification을 할 수 있도록 함
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 50)
model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=lr)

print("Created a learning model and optimizer")

def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def train(model, train_loader, optimizer, epoch):
    model.train()
    for i, (image, target) in enumerate(train_loader):
        image , target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        optimizer.zero_grad()
        one_hot = torch.zeros(target.size(0), 50).to(target.device)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        train_loss = softXEnt(output, one_hot).to(DEVICE)

        train_loss.backward()
        optimizer.step()

        if i % 10 == 0:
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
early_stop = 0
patience = 5

for epoch in range(EPOCH):
    # early_stop
    if early_stop > patience:
        print(f"early_stop epochs : {epoch}")
        break
    
    # 각 에폭마다 따로 학습을 하고 테스트를 하고 다시 이어서 학습
    train_loss = train(model, train_loader, optimizer, epoch)
    val_loss, val_accuracy = evaluate(model, val_loader)

    wandb.log({"val_accuracy": val_accuracy, "val_loss": val_loss})

    # save bast model
    if val_accuracy > best : 
        best = val_accuracy
        torch.save(model.state_dict(), "./best_model.pth")
        early_stop = 0
    else:
        early_stop += 1
        
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