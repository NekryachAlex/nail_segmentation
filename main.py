import Neural_network.UNet as unet_implementation
import Data_proccesing.Dataloader as Dataset
import Metrics.Calculations as metrics_functions

import torchvision.transforms as transforms
import albumentations as A
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from sklearn.model_selection import train_test_split

import os
#Getting path to the config (must be placed into directory with main.py)
script_directory = os.path.dirname(os.path.abspath(__file__))
path_to_config = str(script_directory) + '/config.yaml'
with open(path_to_config, 'r') as file:
    config = yaml.safe_load(file)

#Getting config
divice_ = config['device']
images_path = config['images_path']
labels_path = config['labels_path']
network_config_path = config['network_config_path']
loss_path = config['loss_path']

iou_path = config['iou_path']

device = torch.device(divice_)
num_epochs = config['num_epochs']

#Creating augmentation
trans1 = transforms.Compose([transforms.Grayscale(1), transforms.Resize((224,224)), transforms.ToTensor()])
trans2 = A.Compose([
    A.Flip(p=0.5),
    A.RandomBrightnessContrast(p=0.6),
    A.Sharpen(p = 0.6)
])

#Preparing for training
model = unet_implementation.UNet(n_channels = 1, n_classes = 1)
model.to(device)
loss3 = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

loss_history = []
valid_losses = []
IoU_train = []
IoU_valid = []

#Train process. It includes training and validation datasets, losses and IoU calculation. Each epoch all dataset is considered. 
#Losses and IoU are saved in the same directory where is the main.py.
#Model is also saved.
for epoch in range(num_epochs):
    print(f'epoch {epoch}')
    dataset = Dataset.NailsDataset(images_path, labels_path, trans1 = trans1, trans2 = trans2)
    train, val = train_test_split(dataset, train_size=0.9, shuffle=True, random_state=0)
    train_loader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)
    train_running_loss = 0
    valid_running_loss = 0
    model.train()
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss3(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(loss.item())
        train_running_loss += loss.item()
    for x1, y1 in valid_loader:
        x1, y1 = x1.to(device), y1.to(device)
        with torch.no_grad():
            y1_pred = model(x1)
            loss1 = loss3(y1_pred, y1)
            valid_losses.append(loss1.item())
            valid_running_loss += loss.item()
    train_running_loss = train_running_loss/len(train_loader)
    print("Train loss - ", train_running_loss)
    valid_running_loss = valid_running_loss/len(valid_loader)
    print("Valid loss - ", valid_running_loss)
    IoU_t = metrics_functions.check_accuracy(train_loader, model,device)
    IoU_train.append(IoU_t.item())
    IoU_v = metrics_functions.check_accuracy(valid_loader, model,device)
    IoU_valid.append(IoU_v.item())
fig, axes = plt.subplots(1, 2)
axes[0].plot(range(len(loss_history)), loss_history)
axes[0].set_title("Train Loss")
axes[1].plot(range(len(valid_losses)), valid_losses)
axes[1].set_title("Validation Loss")
fig.set_figwidth(10)   
fig.set_figheight(5)

plt.savefig(loss_path)
plt.show(block=False) 



fig, axes = plt.subplots(1, 2)
axes[0].plot(range(len(IoU_train)), IoU_train)
axes[0].set_title("Train IoU")
axes[1].plot(range(len(IoU_valid)), IoU_valid)
axes[1].set_title("Validation IoU")
fig.set_figwidth(10)   
fig.set_figheight(5) 

plt.savefig(iou_path)
plt.show(block=False)      



torch.save({

    'model_state_dict': model.state_dict(),

}, network_config_path)


