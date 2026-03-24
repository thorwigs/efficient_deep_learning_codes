from ast import mod

import torch
import torchinfo
from torch import nn
import torch.ao.quantization as quant
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
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/ma copine")

import densenet
import densenet_8bits
import test


test_dataloader = test.load_cifar_test(test.load_test_transformation())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_cpt = torch.load('stats/DN_100_ADAM_scheduler_mixup_quant_1.pth')

config2 = {"epochs": 300,
          'lr': 0.1,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}

type_8 = True
if type_8:
    model = densenet_8bits.densenet_cifar_plus_petit(**config2)

    quant_engine = "fbgemm"

    model.qconfig = quant.get_default_qat_qconfig(quant_engine)
    torch.backends.quantized.engine = quant_engine

    quant.prepare_qat(model, inplace=True)
else:
    model = densenet.densenet_cifar_plus_petit(**config2)
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

epochs = 15
acc = 0
nb_acc = 8
amount_s = 0.3
amount_u = 0
lr_in = 0.01
lr_after = 0.001
momentum = 0.9
weight_decay = 1e-04

optimizer_in = optim.SGD(model.parameters(), lr=lr_in, momentum=momentum, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

print("Test whole network on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

summ = torchinfo.summary(model, (1, 3, 32, 32), verbose=0)
w_f = summ.total_params
print(f"total params at first : {w_f}")

model.train()
print(f"training + pruning : {amount_s} structured and {amount_u} unstructured")

growth_rate = model.growth_rate
reduction = config2["red"]
in_seq = 2*growth_rate
nb = 0

prune.l1_unstructured(model.conv1, name='weight', amount=amount_u)

model.conv1.qconfig = quant.get_default_qat_qconfig("fbgemm")

for name, m in model.named_modules():

    if isinstance(m, (densenet.Bottleneck, densenet_8bits.Bottleneck)):
        # print(f"Starting pruning {m} ({name}) with {len(m)} Bottlenecks")
        liste_pruned = []
        for i in range(m.conv1.weight.data.size()[0]):
            importance = torch.sum(torch.abs(m.conv1.weight.data[i, :, :, :])) + torch.sum(torch.abs(m.conv2.weight.data[:, i, :, :]))
            tot = m.conv1.weight.data.shape[1]*m.conv1.weight.data.shape[2]*m.conv1.weight.data.shape[3]
            tot += m.conv2.weight.data.shape[0]*m.conv2.weight.data.shape[2]*m.conv2.weight.data.shape[3]
            liste_pruned.append((importance/tot, i))

        liste_pruned.sort(key=lambda x: x[0], reverse=True)

        in_planes = in_seq
        new_layers = []

        nb_kept_layers = int((1-amount_s)*len(liste_pruned))

        keep_index = liste_pruned[:nb_kept_layers]
        keep_index = [x[1] for x in keep_index]

        old_dict = m.state_dict()

        conv1_size = m.conv1.weight.data.size()
        new_conv1 = nn.Conv2d(conv1_size[1], nb_kept_layers, kernel_size=conv1_size[2], bias=False)
        new_conv1.weight.data.copy_(old_dict['conv1.weight'][keep_index, :, :, :])


        new_bn2 = nn.BatchNorm2d(nb_kept_layers)
        new_bn2.weight.data.copy_(old_dict['bn2.weight'][keep_index])
        new_bn2.bias.data.copy_(old_dict['bn2.bias'][keep_index])
        new_bn2.running_mean.copy_(old_dict['bn2.running_mean'][keep_index])
        new_bn2.running_var.copy_(old_dict['bn2.running_var'][keep_index])
        
        conv2_size = m.conv2.weight.data.size()
        new_conv2 = nn.Conv2d(nb_kept_layers, conv2_size[0], kernel_size=conv2_size[2], padding=1, bias=False)
        new_conv2.weight.data.copy_(old_dict['conv2.weight'][:, keep_index, :, :])

        m.conv1 = new_conv1
        m.bn2 = new_bn2
        m.conv2 = new_conv2

        prune.l1_unstructured(m.conv1, name='weight', amount=amount_u)
        prune.l1_unstructured(m.conv2, name='weight', amount=amount_u)

        m.conv1.qconfig = quant.get_default_qat_qconfig(quant_engine)
        m.bn2.qconfig   = quant.get_default_qat_qconfig(quant_engine)
        m.conv2.qconfig = quant.get_default_qat_qconfig(quant_engine)


        model.to(device)

        acc += 1

        if acc % nb_acc == 0:
            print(f"training {acc}")
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer_in.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_in.step()


    elif isinstance(m, (densenet.Transition, densenet_8bits.Transition)):
        prune.l1_unstructured(m.conv, name='weight', amount=amount_u)
        m.conv.qconfig = quant.get_default_qat_qconfig(quant_engine)
    elif isinstance(m, (nn.Linear)):
        prune.l1_unstructured(m, name='weight', amount=amount_u)
        m.qconfig = quant.get_default_qat_qconfig(quant_engine)


masks = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and hasattr(module, 'weight') and prune.is_pruned(module):
        masks[name] = module.weight_mask.clone()
        prune.remove(module, 'weight')
        module.weight = nn.Parameter(module.weight.data)    


model.qconfig = quant.get_default_qat_qconfig(quant_engine)
quant.prepare_qat(model, inplace=True)

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) and name in masks:
        prune.custom_from_mask(module, name='weight', mask=masks[name])


optimizer_after = optim.SGD(model.parameters(), lr=lr_after, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_after,
    T_max=epochs
)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer_after.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_after.step()

    scheduler.step()
    test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight') and prune.is_pruned(module):
        prune.remove(module, 'weight')

summ = torchinfo.summary(model, (1, 3, 32, 32), verbose=0)
w_e = summ.total_params
print(f"total params at the end : {w_e}")
print(f"pruning rate : {(1-w_e/w_f)*100:0.2f}%")


model.eval()
if type_8:
    device = torch.device("cpu")
    model.to(device)

    for name, module in model.named_modules():
        if isinstance(module, torch.ao.nn.qat.modules.linear.Linear):
            module.weight = nn.Parameter(module.weight.data.cpu())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.cpu())


    quant.convert(model, inplace=True)

print("Test network after fine tunning on cifar test :")
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss()))

path = "stats/DN_pruning_struct_quant_0_3_filterV2"
print(f"Saving model at {path}.pth")
torch.save(model.state_dict(), path+".pth")

if not type_8:
    print("Test network after half:")
    model.half()
    test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss(), half=True))