import torch

import torchinfo


import sys
sys.path.append("/homes/y23charo/Documents/effeicient_deep_learning/codes_lab1/")

from densenet import *
import test

test_dataloader = test.load_cifar_test(test.load_test_transformation())

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#loaded_cpt = torch.load('stats/DN_300_scheduler_mixup_1.pth')

config2 = {"epochs": 300,
          'lr': 0.1,
          "momentum": 0.9,
          "weight_decay": 5e-4, 
          "nb_blocks": [4,8,16,12],
          "gr": 8,
          "red": 0.5}

model = densenet_cifar_plus_petit(**config2)
#model.load_state_dict(loaded_cpt)
model.eval()

#model.to(device)

ps = 0
pu = 0.2
pu_vrai = (1-ps)*pu

qw = 32
qa = 32

summ = torchinfo.summary(model, (1, 3, 32, 32))
w = summ.total_params
f = summ.total_mult_adds

sum_param = ((1-(ps+pu_vrai))*qw/32*w)/5600000
sum_ops   = ((1-ps)*max(qw, qa)/32*f)/280000000

score = sum_param + sum_ops

print(f"===================")
print(f"score : {score:0.3f}")
print(f"sum_param = {sum_param:0.3f}")
print(f"sum_ops   = {sum_ops:0.3f}")
print(f"with : ")
print(f"\tps = {ps:0.3f}")
print(f"\tpu = {pu:0.3f}")
print(f"\tqw = {qw:0.3f}")
print(f"\tqa = {qa:0.3f}")
print(f"\tw  = {w:0.3f}")
print(f"\tf  = {f:0.3f}")
print(f"===================")
