import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader,ConcatDataset,Subset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
from Dataset.CUB2011 import CUB2011
import wandb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print(DEVICE)

# ResNet18 모델의 구조를 정의합니다.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
### Transfer Learning ###
# pre-train된 resnet 뒤에 fc 50개 짜리로 바꾸어서 50개의 class에 대한 classification을 할 수 있도록 함
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 50)
model.to(DEVICE)

# 사전 훈련된 가중치를 로드합니다.
# 'map_location'을 사용하여 가중치를 CPU 또는 GPU로 로드할 수 있습니다.
state_dict = torch.load("./best_model_94.295.pth", map_location='cpu')
model.load_state_dict(state_dict)

# val, test는 augmentation 하지 않음
transforms_valtest = transforms.Compose([
    transforms.CenterCrop((448,448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 모델을 평가 모드로 설정합니다.
model.eval()


target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

test_file = os.listdir("/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/test")
for i in test_file:
    num, label = i.split("_")

    targets = [ClassifierOutputTarget(int(label[:-4]))]

    img_path = "/home/gw6/CV_Active/dataset/CUB_200_2011_repackage_class50/datasets/test/" + i
    img = Image.open(img_path).convert('RGB')
    # 중앙에서 448x448 크기로 크롭하기
    center_crop = transforms.CenterCrop((448, 448))
    img = center_crop(img)


    input_tensor = transforms_valtest(img).unsqueeze(0)

    # CAM 계산
    grayscale_cam = cam(input_tensor=input_tensor.cuda(), targets=targets)

    # CAM 이미지만 가져옵니다 (배치 내 첫 번째 이미지에 대한 CAM).
    grayscale_cam = grayscale_cam[0, :]

    # 원본 이미지를 numpy 배열로 변환
    rgb_img = np.array(img).astype(np.float32)

    # 이미지의 값 범위를 [0, 1]로 조정합니다.
    rgb_img /= 255.0


    # CAM을 원본 이미지 위에 오버레이
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    with torch.no_grad():
        device = next(model.parameters()).device
        image = input_tensor.to(device)
        label = torch.tensor([int(label[:-4])], dtype=torch.int)
        label = label.to(device)

        # 모델을 통과시켜 출력을 얻습니다.
        output = model(image)

        # 최대 확률을 가진 인덱스가 예측입니다.
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(label.view_as(pred)).item() 

        print(f"Number of correct predictions: {correct}")

    # 시각화 이미지를 저장하거나 보여주기
    cv2.imwrite('./result/'+i[:-4] + "_" +str(correct) + ".jpg", visualization)
