import subprocess
import sys

def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"])

install()
print('\n---Additional libraries installation complete---')

import os
import cv2
# from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
            image = self.transform(image=image)["image"]
        return image


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(3, 32, 3, padding=1, stride=2) # stride 2 for homemade maxpool
        self.lin_1 = nn.Linear(64**2*32, 1)
        self.relu = nn.LeakyReLU(0.15)


    def forward(self, images):
        x = self.conv_1(images)
        x = self.relu(x)
        x = x.view(-1, 64**2*32)
        x = self.lin_1(x)
        return x


model = BaselineModel()
model_path = os.path.join(os.getcwd(), 'trained_model/model_weights.pt')
model = model.eval()
model.load_state_dict(torch.load(model_path))
trained_mean = (136.352539062500, 112.022689819336,  96.958969116211)
trained_std = (60.459537506104, 53.603122711182, 53.296249389648)

images_paths = os.listdir(img_dir)
target = torch.Tensor([0 if ')' in filename else 1 for filename in images_paths])

eval_transform = A.Compose([
                    A.Resize(128, 128),
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
    print(f'\nAccuracy = {calculate_accuracy(output, target):.2%}')

[print(p) for p in predicted_paths]



