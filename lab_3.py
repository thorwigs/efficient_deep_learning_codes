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

import sys
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")

from densenet import *
import test

test_dataloader = test.load_cifar_test(test.load_test_transformation())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_cpt = torch.load('stats/DN_pruning_0_2.pth')
pruned = True

config2 = {"epochs": 300,
          'lr': 0.1,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}

model = densenet_cifar_plus_petit(**config2)
if pruned:
    for idx, m in enumerate(model.modules()):
        if hasattr(m, "weight") and isinstance(m, (nn.Linear, nn.Conv2d)):
            prune.random_unstructured(m, name="weight", amount=0)
model.load_state_dict(loaded_cpt)
model.eval()
model.to(device)

print("Test whole network on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))
model.half()
print("Test halfed network on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss(), half=True))
"""


class BC():
    def __init__(self, model):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights
        
        self.target_modules = [] # this will contain the list of modules to be modified

        self.model = model # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save full precision weights
        self.save_params()

        ### (2) Binarize weights using sign function
        for index in range(self.num_of_params):

            w = self.target_modules[index].data

            wbinary = torch.sign(w)
            wbinary[wbinary==0] = 1

            self.target_modules[index].data.copy_(wbinary)
        
    def restore(self):

        ### restore the copy from self.saved_params into the model 

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    def clip(self):

        ## To be completed 
        ## Clip all parameters to the range [-1,1] using Hard Tanh 
        ## you can use the nn.Hardtanh function

        hardtanh = nn.Hardtanh(min_val=-1.0, max_val=1.0)
        for index in range(self.num_of_params):

            w = self.target_modules[index].data
            wclipped = hardtanh(w)
            self.target_modules[index].data.copy_(wclipped)





    def forward(self,x):

        ### This function is used so that the model can be used while training
        out = self.model(x)
        return out
    

model_bc = BC(densenet_cifar())


import wandb

wandb.login()

project = "training quantization"

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

transform_train_DA = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10train_DA = CIFAR10(rootdir,train=True,download=True,transform=transform_train_DA)

trainloader = DataLoader(c10train,batch_size=64,shuffle=True)

from torch.utils.data import default_collate
def collate_fn(batch):
    DA = v2.MixUp(num_classes=10)
    return DA(*default_collate(batch))

trainloader_DA = DataLoader(c10train_DA,batch_size=64,shuffle=True, collate_fn=collate_fn)

import torch.optim as optim

def train(net, train_loader, test_loader, path, run, stats={}, epoch_start=0, optimizer_class=optim.SGD, criterion=nn.CrossEntropyLoss(), epochs=30, lr=0.01, momentum=0.9, weight_decay=5e-4):
    optimizer = optimizer_class(net.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Begining training on device : {device}")
    net.model.to(device)

    for epoch in range(epoch_start, epoch_start + epochs):  # loop over the dataset multiple times
        print(f"Begining epoch {epoch+1}:")

        running_loss = 0.0
        total_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Binarization
            net.binarization()

            # forward + backward + optimize
            outputs = net.model(inputs)


            loss = criterion(outputs, labels)            
            loss.backward()

            net.restore()

            optimizer.step()

            net.clip()
            
            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            n_cumulation = 156 # batches of 32 means every 5000 data
            if i % n_cumulation == n_cumulation-1:    # print every n_cumulation mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / n_cumulation))
                running_loss = 0.0
        
        scheduler.step()

        net.binarization()
        total, corect, test_loss = test.test(net.model, test_loader, device, criterion)
        net.restore()
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
                torch.save(net.model.state_dict(), path+".pth")

    if path != "":
        with open(path+".json", "w") as file:
            json.dump(stats, file)
    
    print(f"Best test accuracy during training : {best_acc}")

    return stats

with wandb.init(project=project, config=config) as run:
    path = "/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/stats/DN_100_scheduler_mixup_quant_2"
    stats = train(model_bc, trainloader, test_dataloader, path, run, **config)
    #stats = train(model_bc, trainloader_DA, test_dataloader, path, run, stats, config["epochs"], **config)


model_bc.binarization()
print("Test binarized model :")
test.read(*test.test(model_bc.model, test_dataloader, device, nn.CrossEntropyLoss()))
"""