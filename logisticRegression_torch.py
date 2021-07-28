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

#PREPARE THE DATA
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

#hyperparameters
BATCH_SIZE=100
n_iters=10001
num_epoch=n_iters/ (len(x_train)/BATCH_SIZE)
num_epoch=int(num_epoch)

#tensor dataset and dataloader

train=TensorDataset(x_train,y_train)
test=TensorDataset(x_test,y_test)

train_loader=DataLoader(train,batch_size=BATCH_SIZE,shuffle=False)
test_loader=DataLoader(test,batch_size=BATCH_SIZE,shuffle=False)


plt.imshow(train_x[10].reshape(28,28))
plt.axis("off")
plt.title(str(train_y[10]))
plt.savefig("graph.png")
plt.show()
#%%
class LogisticRegressionModule(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(LogisticRegressionModule,self).__init__()
        self.linear=nn.Linear(input_dim, output_dim)

    def forward(self , x):
        return self.linear(x)

input_dim=28*28
output_dim=10

model=LogisticRegressionModule(input_dim, output_dim)
error=nn.CrossEntropyLoss()
lr=0.001
optimizer=torch.optim.SGD(params=model.parameters(), lr=lr)

#%%
count=0
loss_list=list()
iter_list=list()

for epoch in range(num_epoch):

    for i, (images,labels) in enumerate(train_loader):
        
        #define variables
        train=Variable(images.view(-1,28*28))
        labels=Variable(labels)

        #clear gradients
        optimizer.zero_grad()

        #forward prob
        outputs=model(train)

        #calculater loss softmax 
        loss=error(outputs,labels)

        #calculate grads
        loss.backward()

        #update parameters
        optimizer.step()
        count+=1

        #prediction

        if count%50 ==0:
            correct = 0
            total = 0

            #precit test dataset
            for images, labels in test_loader:
                
                test=Variable(images.view(-1,28*28))
                outputs=model(test)
                predicted=torch.max(outputs.data,1)[1]
                total+=len(labels)
                correct+=(predicted == labels).sum()

            acc=100*correct/float(total)

            loss_list.append(loss.data)
            iter_list.append(count)

        if count %500== 0 :
            print("Iteration: {} Loss: {} Accuracy {}".format(count,loss.data,acc))

#%%
plt.plot(iter_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression : Loss vs Number of iteration")
plt.show()

