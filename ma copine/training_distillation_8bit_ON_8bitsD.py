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
import torch.optim as optim
import sys
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")
import densenet_8bits
import densenet_8bits_dfactorization
import test
from os.path import exists, dirname, basename


test_dataloader = test.load_cifar_test(test.load_test_transformation())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config2 = {"epochs": 300,
          'lr': 0.001,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}


# J'importe le teacher model
teachermodel = densenet_8bits.densenet_cifar_plus_petit(**config2)
teachermodel.qconfig = quant.get_default_qat_qconfig("fbgemm")
quant.prepare_qat(teachermodel, inplace=True)
loaded_cpt = torch.load('stats/DN_100_ADAM_scheduler_mixup_quant_1.pth')
teachermodel.load_state_dict(loaded_cpt)


# J'importe le student model
studentmodel = densenet_8bits_dfactorization.densenet_cifar_plus_petit(**config2)
studentmodel.qconfig = quant.get_default_qat_qconfig("fbgemm")
torch.backends.quantized.engine = 'fbgemm'
quant.prepare_qat(studentmodel, inplace=True)
loaded_cpt2 = torch.load('stats/DN_100_scheduler_mixup_quant_3.pth')
studentmodel.load_state_dict(loaded_cpt2)

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



def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, soft_target_loss_weight, ce_loss_weight, device,test_loader, criterion=nn.CrossEntropyLoss()):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0
    stats={}

    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    for epoch in range(epochs):
        print(f"Begining epoch {epoch+1}:")
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

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

        total, corect, test_loss = test.test(student, test_loader, device, criterion)
    
        acc = (corect/total*100)
        print(f"End of epoch {epoch+1} : \n\tmean loss = {running_loss/(i+1):0.3f} \n\ttest loss = {test_loss:0.3f} \n\ttest accuracy = {acc:0.2f}%")

        stats[f"epoch {epoch}"] = {"mean loss": running_loss/(i+1),
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
                torch.save(student.state_dict(), to_write+".pth")

    if path != "":
        to_write = path
        if not exists(dirname(path)):
            to_write = "./"+basename(path)
        with open(to_write+".json", "w") as file:
            json.dump(stats, file)
    
    print(f"Best test accuracy during training : {best_acc}")

    return stats

with wandb.init(project=project, config=config2) as run:
    path = "/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/stats/DN_100_scheduler_mixup_distillation_8bitD"
    train_knowledge_distillation(teacher=teachermodel, student=studentmodel, train_loader=trainloader_DA, epochs=10, learning_rate=0.001, T=2, soft_target_loss_weight=0.25, ce_loss_weight=0.75, device=device,test_loader=test_dataloader)
    
studentmodel.eval()
quant.convert(studentmodel, inplace=True)

print("Test quantized model :")
test.read(*test.test(studentmodel, test_dataloader, device, nn.CrossEntropyLoss()))
