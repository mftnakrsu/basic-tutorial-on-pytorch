import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn

path="mnist/"

#first column label , other columns image pixels

#PREPARE THE DATA ANN ve LR ile aynı adımlar

train = pd.read_csv(path+"train.csv",dtype="float32")
train_y=train.label.values
train_x=train.drop('label',axis=1).values/255
#splitting data
x_train,x_test,y_train,y_test=train_test_split(train_x,train_y,test_size=0.2,random_state=42)
#array to tensor
x_train=torch.from_numpy(x_train)
x_test=torch.from_numpy(x_test)
y_train=torch.from_numpy(y_train).type(torch.LongTensor)
y_test=torch.from_numpy(y_test).type(torch.LongTensor)
#tensor dataset and dataloader

BATCH_SIZE=100
n_iters=2500
num_epoch=int(n_iters/ (len(x_train)/BATCH_SIZE))

train=TensorDataset(x_train,y_train)
test=TensorDataset(x_test,y_test)
train_loader=DataLoader(train,batch_size=BATCH_SIZE,shuffle=False)
test_loader=DataLoader(test,batch_size=BATCH_SIZE,shuffle=False)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel,self).__init__()

        #out channel --> filter
        self.cnn1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,stride=1,padding=0)
        self.relu1=nn.ReLU()
        self.maxPool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,stride=1,padding=0)
        self.relu2=nn.ReLU()
        self.maxPool2=nn.MaxPool2d(kernel_size=2)

        #cnn-maxpool-cnn-max pool (28-24-12-8-4)
        self.fc1=nn.Linear(32*4*4, 10) 

    def forward(self,x):
        
        x=self.cnn1(x)
        x=self.relu1(x)
        x=self.maxPool1(x)

        x=self.cnn2(x)
        x=self.relu2(x)
        x=self.maxPool2(x)
        #flatten
        x=x.view(x.size(0),-1)
        x=self.fc1(x)

        return x

model=CNNModel()
error=nn.CrossEntropyLoss()
lr=.002
optimizer=torch.optim.Adam(params=model.parameters(),lr=lr)

#Training step

count=0
loss_list=[]
iter_list=[]
acc_list=[]

for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):
        
        train=Variable(images.view(100,1,28,28))
        labels=Variable(labels)
        
        optimizer.zero_grad()
        
        outputs=model(train)

        loss=error(outputs,labels)

        loss.backward()

        optimizer.step()

        count+=1

        if count%50==0:
            correct=0
            total=0

            for i,(images,labels) in enumerate(test_loader):
                test=Variable(images.view(100,1,28,28))
                outputs=model(test)
                predicted=torch.max(outputs.data,1)[1]
                total+=len(labels)
                correct+=(predicted==labels).sum()

            acc=100*correct/float(total)

            loss_list.append(loss.data)
            iter_list.append(count)
            acc_list.append(acc)

        if count%100==0:
            print("Iteration {} Loss {} Accuracy {}".format(count,loss.data,acc))

#loss & acc visualization 

plt.plot(iter_list,loss_list)
plt.xlabel("Iteratıon")
plt.ylabel("Loss")
plt.title("ANN: Loss vs Number of Iteration")
plt.show()

plt.plot(iter_list,acc_list,color="red")
plt.xlabel("Iteratıon")
plt.ylabel("Accuracy")
plt.title("ANN: Accuracy vs Number of Iteration")

plt.show()