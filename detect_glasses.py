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
from facenet_pytorch import MTCNN
import numpy as np


start_time = time.time()
img_dir = sys.argv[-1]

mtcnn = MTCNN(128, margin=8)
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
        image = mtcnn(image)
        image = torch.swapaxes(image, 0, 2)
        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]
        return image


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1, stride=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, stride=2)
        )

        self.decoder = nn.Sequential(
            # nn.Dropout2d(p=0.3),
            nn.Linear(32**2*128, 1)
        )

    def forward(self, images):
        x = self.layer(images)
        x = x.view(-1, 32**2*128)
        x = self.decoder(x)
        return x


model = DetectionModel()
model_path = os.path.join(os.getcwd(), 'trained_model/model_weights.pt')
model = model.eval()
model.load_state_dict(torch.load(model_path))
trained_mean = (136.352539062500, 112.022689819336,  96.958969116211)
trained_std = (60.459537506104, 53.603122711182, 53.296249389648)

images_paths = os.listdir(img_dir)
target = torch.Tensor([0 if ')' in filename else 1 for filename in images_paths])


# заменить на torchvsion - меньше импортов
# перейти с ргб на grayscale
eval_transform = A.Compose([
                    A.Normalize(mean=trained_mean, std=trained_std),
                    ToTensorV2(),
])

images_dataset = GlassesDataset(images_filepaths=images_paths, transform=eval_transform)
image_loader = DataLoader(images_dataset, batch_size=128, shuffle=False, pin_memory=True)

def calculate_accuracy(predictions, target):
    predictions = torch.sigmoid(predictions) >= 0.5
    target = target.view(-1, 1) == 1
    return torch.true_divide((target == predictions).sum(dim=0), output.size(0)).item()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

predicted_paths = []
with torch.no_grad():
    for images in image_loader:
        if device.type == 'cuda':
            images = images.to(device, non_blocking=True)
        output = model(images)
        predictions = (torch.sigmoid(output) >= 0.5)[:, 0]
        predicted_paths += [images_paths[i] for i, gls_detected in enumerate(predictions) if gls_detected]
    print(f'\n---Accuracy = {calculate_accuracy(output, target):.2%}---\n')

[print(p) for p in predicted_paths]
print(f'\n---Inference time: {(time.time() - start_time) / len(images_paths) * 1000:.2f} milliseconds per image ---\n')



