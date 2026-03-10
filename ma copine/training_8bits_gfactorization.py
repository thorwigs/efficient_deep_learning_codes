import torch
from torch import nn
import numpy
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
import json
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.ao.quantization as quant
from os.path import exists, dirname, basename


import sys
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")

from densenet_8bits_gfactorization import *
import test

test_dataloader = test.load_cifar_test(test.load_test_transformation())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config2 = {"epochs": 300,
          'lr': 0.001,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}


net = densenet_cifar_plus_petit(**config2)

net.qconfig = quant.get_default_qat_qconfig("fbgemm")
torch.backends.quantized.engine = 'fbgemm'

quant.prepare_qat(net, inplace=True)

import wandb

wandb.login()

project = "training factorization+quantization"


## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only

transform_train_DA = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train_DA = CIFAR10(rootdir,train=True,download=True,transform=transform_train_DA)

from torch.utils.data import default_collate
def collate_fn(batch):
    DA = v2.MixUp(num_classes=10)
    return DA(*default_collate(batch))

trainloader_DA = DataLoader(c10train_DA,batch_size=64,shuffle=True, collate_fn=collate_fn)

import torch.optim as optim

def train(net, train_loader, test_loader, path, run, stats={}, epoch_start=0, optimizer_class="Adam", criterion=nn.CrossEntropyLoss(), epochs=30, lr=0.01, momentum=0.9, weight_decay=5e-4, **argv):
    if optimizer_class == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

        total, corect, test_loss = test.test(net, test_loader, device, criterion)
       
        acc = (corect/total*100)
        print(f"End of epoch {epoch+1} : \n\tmean loss = {total_loss/(i+1):0.3f} \n\ttest loss = {test_loss:0.3f} \n\ttest accuracy = {acc:0.2f}%")

        stats[f"epoch {epoch}"] = {"mean loss": total_loss/(i+1),
                        "test loss": test_loss,
                        "test accuracy": acc,
                        "lr": optimizer.param_groups[0]["lr"]}
        run.log(stats[f"epoch {epoch}"])

        if acc > best_acc:
            best_acc = acc
    
            if path != "":
                to_write = path
                if not exists(dirname(path)):
                    to_write = "./"+basename(path)
                torch.save(net.state_dict(), to_write+".pth")

    if path != "":
        to_write = path
        if not exists(dirname(path)):
            to_write = "./"+basename(path)
        with open(to_write+".json", "w") as file:
            json.dump(stats, file)
    
    print(f"Best test accuracy during training : {best_acc}")

    return stats

with wandb.init(project=project, config=config2) as run:
    path = "/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/stats/DN_100_scheduler_mixup_quant_G"
    stats = train(net, trainloader_DA, test_dataloader, path, run, **config2)


net.eval()
quant.convert(net, inplace=True)

print("Test quantized model :")
test.read(*test.test(net, test_dataloader, device, nn.CrossEntropyLoss()))