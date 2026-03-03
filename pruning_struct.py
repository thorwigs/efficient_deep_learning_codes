import torch
import torchinfo
from torch import nn
# import torch.nn.utils.prune as prune
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

epochs = 100
acc = 0
nb_acc = 1
amount = 0.1
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

summ = torchinfo.summary(model, (1, 3, 32, 32), verbose=0)
w_f = summ.total_params
print(f"total params at first : {w_f}")

model.train()
print(f"training + pruning : {amount}")

growth_rate = model.growth_rate
reduction = config2["red"]
in_seq = 2*growth_rate
nb = 0
for name, m in model.named_modules():

    if isinstance(m, nn.Sequential):
        # print(f"Starting pruning {m} ({name}) with {len(m)} Bottlenecks")
        liste_pruned = []
        nb_bn = 0
        for idx, m2 in enumerate(m.modules()):
            if isinstance(m2, Bottleneck):
                importance = torch.sum(torch.abs(m2.conv2.weight.data))/(m2.conv2.weight.data.shape[0]*m2.conv2.weight.data.shape[1]*m2.conv2.weight.data.shape[2]*m2.conv2.weight.data.shape[3])
                liste_pruned.append((importance, nb_bn))
                nb_bn += 1

        liste_pruned.sort(key=lambda x: x[0])

        in_planes = in_seq
        new_layers = []

        keep_index = liste_pruned[:int((1-amount)*len(liste_pruned))]
        keep_index.sort(key=lambda x: x[1])

        for i, (_, idx) in enumerate(keep_index):

            old_layer = m[idx]
            new_layer = Bottleneck(in_planes, growth_rate).to(device)
            old_dict = old_layer.state_dict()
            # print(in_planes, idx, i, keep_index)

            liste_index = list(range(idx*growth_rate, idx*growth_rate + in_seq))

            keeping = list(set([(idx-i_keep-1)*growth_rate + j for _, i_keep in keep_index[:i] for j in range(growth_rate)] + liste_index))

            # print(liste_index, keeping, new_layer.bn1.weight.data.shape)

            new_layer.bn1.weight.data.copy_(old_dict['bn1.weight'][keeping])
            new_layer.bn1.bias.data.copy_(old_dict['bn1.bias'][keeping])
            new_layer.bn1.running_mean.copy_(old_dict['bn1.running_mean'][keeping])
            new_layer.bn1.running_var.copy_(old_dict['bn1.running_var'][keeping])

            new_layer.conv1.weight.data.copy_(old_dict['conv1.weight'][:, keeping, :, :])
            
            # same
            new_layer.bn2.weight.data.copy_(old_dict['bn2.weight'])
            new_layer.bn2.bias.data.copy_(old_dict['bn2.bias'])
            new_layer.bn2.running_mean.copy_(old_dict['bn2.running_mean'])
            new_layer.bn2.running_var.copy_(old_dict['bn2.running_var'])
            new_layer.conv2.weight.data.copy_(old_dict['conv2.weight'])

            new_layers.append(new_layer)
            in_planes += growth_rate

        m = nn.Sequential(*new_layers)
        model._modules[name] = m
        model.to(device)


        # Transitions
        nb += 1
        if nb < 4:
            # print("TRANSITION", nb)
            trans = getattr(model, "trans"+str(nb))

            out_planes = int(math.floor(in_planes*reduction))
            new_trans = Transition(in_planes, out_planes).to(device)
            # print(trans, new_trans)

            idx = nb_bn

            liste_index = list(range(idx*growth_rate, idx*growth_rate + in_seq))
            keeping = list(set([(idx-i_keep-1)*growth_rate + j for _,i_keep in keep_index for j in range(growth_rate)] + liste_index))

            # print(keeping, idx, keep_index, trans.bn.weight.data.shape)
            new_trans.bn.weight.data.copy_(trans.bn.weight.data[keeping])
            new_trans.bn.bias.data.copy_(trans.bn.bias.data[keeping])
            new_trans.bn.running_mean.copy_(trans.bn.running_mean[keeping])
            new_trans.bn.running_var.copy_(trans.bn.running_var[keeping])

            importance = torch.sum(torch.abs(trans.conv.weight.data), dim=[1,2,3])
            best_index = torch.sort(importance, dim=0, descending=True)[1][:out_planes]

            # print(liste_index, keeping, best_index, trans.conv.weight.data.shape)

            weights_temp = trans.conv.weight.data[best_index, :, :, :]
            
            # print(weights_temp.shape, new_trans.conv.weight.data.shape, keeping)
            weights_temp = weights_temp[:, keeping, :, :]
            new_trans.conv.weight.data.copy_(weights_temp)

            setattr(model, "trans"+str(nb), new_trans)

            in_seq = out_planes
        else:
            new_bn = nn.BatchNorm2d(in_planes).to(device)
            new_linear = nn.Linear(in_planes, model.linear.out_features).to(device)
            
            idx = nb_bn

            liste_index = list(range(idx*growth_rate, idx*growth_rate + in_seq))

            keeping = list(set([(idx-i_keep-1)*growth_rate + j for _,i_keep in keep_index for j in range(growth_rate)] + liste_index))

            new_bn.weight.data.copy_(model.bn.weight.data[keeping])
            new_bn.bias.data.copy_(model.bn.bias.data[keeping])
            new_bn.running_mean.copy_(model.bn.running_mean[keeping])
            new_bn.running_var.copy_(model.bn.running_var[keeping])

            new_linear.weight.data.copy_(model.linear.weight.data[:, keeping])
            new_linear.bias.data.copy_(model.linear.bias.data)

            setattr(model, "bn", new_bn)
            setattr(model, "linear", new_linear)
            in_seq = in_planes


        # acc += 1

        # if acc % nb_acc == 0:
        #     print(f"training {acc}")
        #     for i, data in enumerate(trainloader, 0):
        #         # get the inputs; data is a list of [inputs, labels]
        #         inputs, labels = data[0].to(device), data[1].to(device)

        #         # zero the parameter gradients
        #         optimizer.zero_grad()

        #         # forward + backward + optimize
        #         outputs = model(inputs)

        #         loss = criterion(outputs, labels)
        #         loss.backward()
        #         optimizer.step()


summ = torchinfo.summary(model, (1, 3, 32, 32), verbose=0)
w_e = summ.total_params
print(f"total params at the end : {w_e}")
print(f"pruning rate : {(1-w_e/w_f)*100:0.2f}%")


for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
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

path = "stats/DN_pruning_struct_0_05"
torch.save(model.state_dict(), path+".pth")

print("Test network after half:")
model.half()
test.read(*test.test(model, test_dataloader, device, nn.CrossEntropyLoss(), half=True))