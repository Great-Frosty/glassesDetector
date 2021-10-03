import subprocess
import sys

def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

install()
print('\n---Additional libraries installation complete---')

import time
import os
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision import models

from facenet_pytorch import MTCNN

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
        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]
        return image


model = getattr(models, 'mobilenet_v3_small')(pretrained=True,)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=48, out_features=1, bias=True),

)
model.features = nn.Sequential(*list(model.children())[-3][:8])

model_path = os.path.join(os.getcwd(), 'trained_model/mobnet_weights.pt')
model.load_state_dict(torch.load(model_path))
model = model.eval()

images_paths = os.listdir(img_dir)
target = torch.Tensor([0 if ')' in filename else 1 for filename in images_paths])
eval_transform = A.Compose([
                    A.SmallestMaxSize(max_size=300),
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

inf_time = time.time() - start_time
[print(path) for path in predicted_paths]
print(f'\n---Inference time: {(inf_time) / len(images_paths) * 1000:.2f} milliseconds per image ---\n')



