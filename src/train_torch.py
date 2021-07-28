import os
from matplotlib import transforms
import torch
import torchvision
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

train_dir = r"intel_images/seg_train"
val_dir = r"intel_images/seg_test"

num_of_classes = len(os.listdir(train_dir))

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((150, 150)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(())
])

train_data = torchvision.datasets.ImageFolder(train_dir, transform=transform)
val_data = torchvision.datasets.ImageFolder(val_dir, transform=transform)

valid_size = 0.15

train_loader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True, num_workers=2)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=50, shuffle=True, num_workers=2)

