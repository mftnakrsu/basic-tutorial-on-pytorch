import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader


#TRANSFORMS OR AUGMENTATION
transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#dataset download

trainset=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
testset=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform)

BATCH_SIZE=4

trainloader=DataLoader(trainset,batch_size=BATCH_SIZE)
testloader=DataLoader(testset,batch_size=BATCH_SIZE)

classes=("plane","car","bird","cat","deer","dog","frog","horse","ship","truck")

def imshow(img):
    img=img/2+ 0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

dataiter=iter(trainloader)
images,labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        self.conv1=nn.Conv2d(3, 6, 5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6, 16, 5)

        self.fc1=nn.Linear(16*5*5, 120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)   

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

use_gpu=True

if use_gpu:
    device= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.is_available():
        net=Net().to(device)
        print("gpu avaliable")
        print(torch.cuda.get_device_name())
    else:
        net=Net()
        print("cpu")

error=nn.CrossEntropyLoss()
optimizer=optim.SGD(net.parameters(), lr=0.001,momentum=0.8)

for epoch in range(10):
    running_loss=0.0

    for i,data in enumerate(trainloader,0):
        inputs,labels=data

        if use_gpu:
            if torch.cuda.is_available():
                inputs,labels = inputs.to(device),labels.to(device)
        
        optimizer.zero_grad()
        outputs=net(inputs)
        loss=error(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

        if i%2000==1999:
            print( "[%d,%5d] loss: %.3f" % (epoch+1, i+1,running_loss/2000))
            running_loss=0.0

print("Training is done")