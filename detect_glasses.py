import subprocess
import sys

def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

install()
print('\n---Additional libraries installation complete---')

import time
import os
import cv2
# from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision.models as models


img_dir = sys.argv[-1]

class GlassesDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(os.path.join(img_dir, image_filepath))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = torch.swapaxes(image, 0, 2)
        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]
        return image


# class ConvModel2(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()

    #     self.layer = nn.Sequential(
    #         nn.Conv2d(3,12,3, padding=1, stride=1),
    #         nn.BatchNorm2d(12),
    #         nn.ReLU(),
    #         nn.MaxPool2d(4, stride=4), # 66
    #         nn.Conv2d(12,24,3, padding=1, stride=1),
    #         nn.BatchNorm2d(24),
    #         nn.ReLU(),
    #         nn.MaxPool2d(4, stride=4), # 33
    #         nn.Conv2d(24,48,3, padding=1, stride=1),
    #         nn.BatchNorm2d(48),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2), # 16
    #         nn.Conv2d(48,96,3, padding=1, stride=1),
    #         nn.BatchNorm2d(96),
    #         nn.ReLU(),
    #         nn.AdaptiveMaxPool2d(4),
    #     )

    #     self.classifier = nn.Sequential(
    #         nn.Linear(4**2*96, 1)
    #     )

    # def forward(self, images):
    #     x = self.layer(images)
    #     x = x.view(-1, 4**2*96)
    #     x = self.classifier(x)
    #     return x


model = getattr(models, 'mobilenet_v3_small')(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(in_features=576, out_features=1024, bias=True),
    nn.Hardswish(),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1024, out_features=1, bias=True),
)
model_path = os.path.join(os.getcwd(), 'trained_model/transfer_weights.pt')
model = model.eval()
model.load_state_dict(torch.load(model_path))

images_paths = os.listdir(img_dir)
target = torch.Tensor([0 if ')' in filename else 1 for filename in images_paths])


# заменить на torchvsion - меньше импортов
# перейти с ргб на grayscale
eval_transform = A.Compose([
                    A.SmallestMaxSize(max_size=230),
                    A.CenterCrop(height=224, width=224),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
])

images_dataset = GlassesDataset(images_filepaths=images_paths, transform=eval_transform)
image_loader = DataLoader(images_dataset, batch_size=128, shuffle=False, pin_memory=True)

def calculate_accuracy(predictions, target):
    predictions = torch.sigmoid(predictions) >= 0.5
    target = target.view(-1, 1) == 1
    return torch.true_divide((target == predictions).sum(dim=0), output.size(0)).item()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device, non_blocking=True)
# device = torch.device('cpu')
start_time = time.time()

predicted_paths = []
with torch.no_grad():
    for images in image_loader:
        if device.type == 'cuda':
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
        output = model(images)
        predictions = (torch.sigmoid(output) >= 0.5)[:, 0]
        predicted_paths += [images_paths[i] for i, gls_detected in enumerate(predictions) if gls_detected]
    print(f'\n---Accuracy = {calculate_accuracy(output, target):.2%}---\n')

[print(p) for p in predicted_paths]
print(f'\n---Inference time: {(time.time() - start_time) / len(images_paths) * 1000:.2f} milliseconds per image ---\n')



