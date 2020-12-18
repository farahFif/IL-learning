from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import Dataset as ds
from torch.utils.data import DataLoader
from Mobilenet import MobileNetV2
from Inc_learning import *
from aLoader import aLoader
from Dataset import CIFAR10, CIFAR100
from sklearn.utils import class_weight
from train_test import Train_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Epochs = 50
net = MobileNetV2()
net = net.to(device)

dataset = CIFAR100()
inc_learn = Incremental_setting(dataset,nb_cl_phase=10)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001 )
                    #    /, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# 5 phases
accuracies = []

training = Train_test(net,optimizer,criterion,device,Epochs)

for i in range(20):
    loader_train , loader_test = inc_learn.get_approp_data(i)
    dd = aLoader(loader_train,transform=dataset.train_transform)
    train_loader = torch.utils.data.DataLoader(dd,
                                                batch_size=64,
                                                shuffle=True,num_workers=16)
    tt = aLoader(loader_test)
    test_loader = torch.utils.data.DataLoader(tt,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=16)

    for epoch in range(0,30):
        fina_acc = training.train(train_loader,epoch)
        test_acc = training.test(train_loader,epoch)

