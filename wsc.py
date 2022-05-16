import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os


classes = ['Angle', 'Box', 'Circle', 'Closeup', 'Crowd', 'Other']

transform = transforms.Compose([transforms.Resize([720, 1280]), transforms.ToTensor()])
training_dataset = torchvision.datasets.ImageFolder(root='./train/', transform=transform)
training_loader = DataLoader(training_dataset, batch_size=len(training_dataset))

# Calculate the mean and std
dataiter = iter(training_loader)
images, _ = dataiter.next()
imgs = images.view(images.size(0), images.size(1), -1)
mean = imgs.mean(2).sum(0) / imgs.size(0)
std = imgs.std(2).sum(0) / imgs.size(0)

# Normalized dataset
normalized_transform = transforms.Compose([
    transforms.Resize([720, 1280]),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])

batch_size = 4
normalized_training_dataset = torchvision.datasets.ImageFolder(root='./train/', transform=normalized_transform)
normalized_testing_dataset = torchvision.datasets.ImageFolder(root='./test/', transform=normalized_transform)
normalized_training_loader = DataLoader(normalized_training_dataset, batch_size=batch_size, shuffle=True)
normalized_testing_loader = DataLoader(normalized_testing_dataset, batch_size=batch_size, shuffle=False)
