from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
from torch.utils.data.dataloader import DataLoader

## Normalization adapted for CIFAR10
def load_test_transformation():
    normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize_scratch,
    ])
    return transform_test


def load_cifar_test(transform):
    ### The data from CIFAR10 are already downloaded in the following folder
    rootdir = '/opt/img/effdl-cifar10/'

    c10test = CIFAR10(rootdir,train=False,download=True,transform=transform)

    testloader = DataLoader(c10test,batch_size=32)

    return testloader

def test(net, test_set, device, criterion, half=False):
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():  # torch.no_grad for TESTING
        for data in test_set:
            inputs, labels = data[0].to(device), data[1].to(device)
            if half:
                inputs = inputs.half()

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item() / labels.size(0)
    return total, correct, total_loss

def read(total, correct, total_loss):
    acc = correct/total*100
    print(f"loss = {total_loss:0.3f}\naccuracy = {acc:0.2f}%")