from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
import json
from torch.utils.data.dataloader import DataLoader
import torch.backends.cudnn as cudnn

import wandb

wandb.login()

project = "training efficient deep learning"

config = {"epochs": 100,
          'lr': 0.1,
          "momentum": 0.9,
          "weight_decay": 5e-4}

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

transform_train_cutmix = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10train_cutmix = CIFAR10(rootdir,train=True,download=True,transform=transform_train_cutmix)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=64,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)

from torch.utils.data import default_collate
def collate_fn(batch):
    cutmix = v2.CutMix(num_classes=10)
    return cutmix(*default_collate(batch))

trainloader_cutmix = DataLoader(c10train_cutmix,batch_size=64,shuffle=True, collate_fn=collate_fn)


## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.


'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def test(net, test_set, device, criterion):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in test_set:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item() / labels.size(0)
    return total, correct, total_loss


import torch.optim as optim

def train(net, train_loader, test_loader, path, run, stats={}, epoch_start=0, optimizer_class=optim.SGD, criterion=nn.CrossEntropyLoss(), epochs=30, lr=0.01, momentum=0.9, weight_decay=5e-4):
    optimizer = optimizer_class(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[epochs//3, epochs*2//3],
        gamma=0.1
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    print(f"Begining training on device : {device}")
    net.to(device)

    for epoch in range(epoch_start, epoch_start + epochs):  # loop over the dataset multiple times
        print(f"Begining epoch {epoch+1}:")

        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            n_cumulation = 156 # batches of 32 means every 5000 data
            if i % n_cumulation == n_cumulation-1:    # print every n_cumulation mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / n_cumulation))
                running_loss = 0.0

        scheduler.step()

        total, corect, test_loss = test(net, test_loader, device, criterion)
        print(f"End of epoch {epoch+1} : \n\tmean loss = {total_loss/(i+1):0.3f} \n\ttest loss = {test_loss:0.3f} \n\ttest accuracy = {(corect/total*100):0.2f}%")

        stats[f"epoch {epoch}"] = {"mean loss": total_loss/(i+1),
                        "test loss": test_loss,
                        "test accuracy": corect/total*100,
                        "lr": optimizer.param_groups[0]["lr"]}
        run.log(stats[f"epoch {epoch}"])

    if path != "":
        torch.save(net.state_dict(), path+".pth")
        with open(path+".json", "w") as file:
            json.dump(stats, file)
    
    return stats

with wandb.init(project=project, config=config) as run:
    net = VGG("VGG19")
    path = "/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/stats/VGG19_100_scheduler_1"
    stats = train(net, trainloader, testloader, path, run, **config)