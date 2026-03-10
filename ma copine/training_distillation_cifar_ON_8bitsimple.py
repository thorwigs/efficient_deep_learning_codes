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


# import sys
# sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")

from densenet_8bits import *
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


# J'importe le student model
studentmodel = densenet_cifar_plus_petit(**config2)
studentmodel.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(studentmodel, inplace=True)
loaded_cpt = torch.load('stats/DN_100_ADAM_scheduler_mixup_quant_1.pth')
studentmodel.load_state_dict(loaded_cpt)


# J'importe le teacher model
studentmodel = densenet_cifar_plus_petit(**config2)

loaded_cpt = torch.load('stats/DN_100_ADAM_scheduler_mixup_quant_1.pth')
studentmodel.load_state_dict(loaded_cpt)

# J'importe les modules wandb
import wandb
wandb.login()
project = "training distillation"


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



def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            #Soften the student logits by applying softmax first and log() second
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

            # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

            # Calculate the true label loss
            label_loss = ce_loss(student_logits, labels)

            # Weighted sum of the two losses
            loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Apply ``train_knowledge_distillation`` with a temperature of 2. Arbitrarily set the weights to 0.75 for CE and 0.25 for distillation loss.
train_knowledge_distillation(teacher=nn_deep, student=studentmodel, train_loader=train_loader, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device)
test_accuracy_light_ce_and_kd = test(studentmodel, test_loader, device)

# Compare the student test accuracy with and without the teacher, after distillation
print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
print(f"Student accuracy without teacher: {test_accuracy_light_ce:.2f}%")
print(f"Student accuracy with CE + KD: {test_accuracy_light_ce_and_kd:.2f}%")