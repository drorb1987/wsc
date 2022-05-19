import torch
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import glob
import os
from mobilenet_v2 import MobileNetV2  
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

CLASSES = ['Angle', 'Box', 'Circle', 'Closeup', 'Crowd', 'Other']

transform = transforms.Compose([transforms.Resize([720, 1280]), transforms.ToTensor()])
training_dataset = torchvision.datasets.ImageFolder(root='./train/', transform=transform)
training_loader = DataLoader(training_dataset, batch_size=len(training_dataset))

# Calculate the mean and std
dataiter = iter(training_loader)
images, _ = dataiter.next()
imgs = images.view(images.size(0), images.size(1), -1)
mean = imgs.mean(2).sum(0) / imgs.size(0)
std = imgs.std(2).sum(0) / imgs.size(0)

# mean = torch.tensor([0.3827, 0.4044, 0.3235])
# std = torch.tensor([0.1845, 0.1861, 0.1851])

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


# Model
model = MobileNetV2(num_classes=len(CLASSES))
model.cuda()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.
    
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(normalized_training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs.cuda()
        labels.cuda()
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        # Make predictions for this batch
        outputs = model(inputs)
        
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        
        # Gather data and report
        running_loss += loss.item()
        if i % 5 == 4:
            last_loss = running_loss / 5 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.
        inputs.detach()
        labels.detach()
            
    return last_loss


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)
    
    # We don't need gradients on to do reporting
    model.train(False)
    
    running_vloss = 0.0
    for i, vdata in enumerate(normalized_testing_loader):
        vinputs, vlabels = vdata
        vinputs.cuda()
        vlabels.cuda()
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
        vinputs.detach()
        vlabels.detach
    
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
    
    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)
    
    epoch_number += 1