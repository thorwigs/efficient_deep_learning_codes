import torch
from torch import nn
import torch.nn.utils.prune as prune
# import numpy
import torchvision.transforms as transforms
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
# import json
# import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim


import sys
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")

from densenet import *
import test

test_dataloader = test.load_cifar_test(test.load_test_transformation())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_cpt = torch.load('stats/DN_300_scheduler_mixup_1.pth')

config2 = {"epochs": 300,
          'lr': 0.1,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}

model = densenet_cifar_plus_petit(**config2)
model.load_state_dict(loaded_cpt)
model.eval()

model.to(device)

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    v2.RandomGrayscale(),
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
trainloader = DataLoader(c10train,batch_size=64,shuffle=True)

epochs = 5
acc = 0
nb_acc = 8
amount = 0.2
lr = 0.01
momentum = 0.9
weight_decay = 5e-04

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)

print("Test whole network on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

model.train()
print(f"traing + pruning : {amount}")
for idx, m in enumerate(model.modules()):
    # print(idx, '->', m)
    if hasattr(m, "weight") and isinstance(m, (nn.Linear, nn.Conv2d)):
        prune.l1_unstructured(m, name="weight", amount=amount)
        # prune.remove(m, name="weight")

        # print("Test pruned network on cifar test :")
        # test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))


        # print("Training")
        acc += 1

        if acc % nb_acc == 0:
            print(f"training {acc}")
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

for epoch in range(epochs):
    # print(f"Epoch {epoch+1}")
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()
    test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

print("Test network after fine tunning on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

path = "stats/DN_pruning_0_2"
torch.save(model.state_dict(), path+".pth")