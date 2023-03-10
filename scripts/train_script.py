#!/usr/bin/env python
# coding: utf-8

# # 455 Final Version 2
# This version uses a dataset with larger images for birds (Rhode Island backyard birds), and resizes images to 128x128


## NOTE: this script is generated at the end of train_script.ipynb. Modifications that should be persisted should be made to
# train_script.ipynb, not this file.

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Datasets
dir_local_dataset = "../dataset"
dir_training = "../models"


# In[3]:


train_split = 0.9

data_classes = np.array(["bird", "squirrel"])

def load_dataset(batch_size, train_split=0.8, val_split=0.1, rand_transform=False):
  # Test_split will be the difference between train_split+val_split and 1

  # Load the dataset and return train, test loaders
  data_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
  ])

  if rand_transform: # Pad, flip, and crop
    train_transform = transforms.Compose([
      transforms.Resize((128,128)),
      transforms.RandomCrop(128, 16),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ])
  else:
    train_transform = data_transform
    
  # todo: implement train&test transform

  dataset = torchvision.datasets.ImageFolder(root=dir_local_dataset, transform=data_transform)

  n_train = int(train_split * len(dataset))
  n_val = int(val_split * len(dataset))
  n_test = len(dataset) - n_train - n_val

  train_dataset, val_dataset, test_dataset =  random_split(dataset, [n_train, n_val, n_test])
    
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return (train_loader, val_loader, test_loader)

# # Model

# In[4]:


class SquirrelNet(nn.Module):
  def __init__(self):
    super(SquirrelNet, self).__init__()

    # conv: |output| = (|input| + 2padding - |filter|)/stride + 1

    # (32+2*1-3)/1 + 1 = 31+1 = 32

    self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)     # (3,32,32)->(16,32,32)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False)    # (16,32,32)->(32,32,32)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)    # (32,32,32)->(64,32,32)
    self.bn3 = nn.BatchNorm2d(64)
    self.conv4 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)   # (64,16,16)->(128,16,16)
    self.bn4 = nn.BatchNorm2d(128)
    self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)   # (64,16,16)->(128,16,16)
    self.bn5 = nn.BatchNorm2d(256)
    self.conv6 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)   # (64,16,16)->(128,16,16)
    self.bn6 = nn.BatchNorm2d(512)
    self.fc1 = nn.Linear(512, 2)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2, stride=2)
    x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), kernel_size=2, stride=2)
    x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), kernel_size=2, stride=2)
    x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), kernel_size=2, stride=2)
    x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), kernel_size=2, stride=2)
    x = F.max_pool2d(F.relu(self.bn6(self.conv6(x))), kernel_size=4)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    return x

# create new model
model = SquirrelNet()


# In[64]:


# load existing model
modelNum = 2
state = torch.load(dir_training + 'model-%d.pkl'%(modelNum))
model.load_state_dict(state['net'])


# # Training

# In[5]:


# setup dataloaders
BATCH_SIZE = 128
train_loader, val_loader, test_loader = load_dataset(BATCH_SIZE)

# from tutorial: show some dataset images
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images[:8]
labels = labels[:8]

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

imshow(torchvision.utils.make_grid(images))
print("Labels:" + ' '.join('%11s' % data_classes[labels[j]] for j in range(8)))


# In[6]:


LR = 0.01
MOMENTUM = 0.9
DECAY = 0.0005
EPOCHS = 20
BATCH_SIZE = 128


# In[7]:


def accuracy(mode, loader):
  # Adapted from tutorial2
  correct = 0
  total = 0
  with torch.no_grad():
        for i, data in enumerate(loader, 0):
          images, labels = data[0].to(device), data[1].to(device)
          outputs = mode(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total


# In[8]:


def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, momentum=MOMENTUM, weight_decay=DECAY):
  model.to(device)

  # set up loss and optimization
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

  training_losses = []

  for epoch in range(epochs):
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
      inputs, labels = data[0].to(device), data[1].to(device)

      optimizer.zero_grad()

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      training_losses.append(loss.item())

      if i % 20 == 19:
          print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss/20))
          running_loss = 0.0
    print("Epoch %d. Validation accuracy=%f" %(epoch+1, accuracy(model, val_loader)))

  return training_losses


# In[9]:


print("Number of model parameters:",  sum(p.numel() for p in model.parameters()))


# In[ ]:


# losses = train(model, train_loader, val_loader, lr=0.1, epochs=15, weight_decay=0)
# losses = train(model, train_loader, val_loader, lr=0.01, epochs=15, weight_decay=0.0001)
# losses = train(model, train_loader, val_loader, lr=0.005, epochs=10, weight_decay=0.0005)
# losses = train(model, train_loader, val_loader, lr=0.001, epochs=10, weight_decay=0.0001)


# In[10]:


losses = train(model, train_loader, val_loader, lr=0.01, epochs=10, weight_decay=0.000)


# In[11]:


print("Train accuracy: %f. Test accuracy: %f" %(accuracy(model, train_loader), accuracy(model, test_loader)))


# In[73]:


# save the model
# modelNum = 2
# state = {'net': model.state_dict()}
# torch.save(state, dir_training + 'model-%d.pkl'%(modelNum))



