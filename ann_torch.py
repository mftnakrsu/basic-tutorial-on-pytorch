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

#data visualization
"""
plt.imshow(train_x[10].reshape(28,28))
plt.axis("off")
plt.title(str(train_y[10]))
plt.savefig("graph.png")
plt.show()"""

class ANNModel(nn.Module):
    #init ve forward: initte sadece tanımlanır 
    # 784-150 relu-150-tanh 150-elu 10 initilaze etti
    #forward'a gidip birbirine baglanacak 
    
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(ANNModel,self).__init__()
        #linear func 1 : input_dim -> hidden , 784-150
        self.fc1=nn.Linear(input_dim, hidden_dim)
        #nonlinearty 1
        self.relu1=nn.ReLU()

        #linear func 2 : hidden -> hidden , 150-150
        self.fc2=nn.Linear(hidden_dim, hidden_dim)
        #nnreality2
        self.tanh2=nn.Tanh()

        #linear func 3 : hidden -> hidden , 150-150
        self.fc3=nn.Linear(hidden_dim, hidden_dim)
        #nnreality3
        self.elu3=nn.ELU()

        #linear func 3 : hidden -> out , 150-10
        self.fc4=nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        out=self.fc1(x)
        out=self.relu1(out)
        out=self.fc2(out)
        out=self.tanh2(out)
        out=self.fc3(out)
        out=self.elu3(out)
        out=self.fc4(out)

        return out
        

input_dim=28*28
hidden_dim=150
output_dim=10

model=ANNModel((input_dim), hidden_dim, output_dim)
error=nn.CrossEntropyLoss()
lr=0.002
optimizer=torch.optim.SGD(params=model.parameters(), lr=lr)

count=0
loss_list=[]
iter_list=[]
acc_list=[]

for epoch in range(num_epoch):
    for i, (images,labels) in enumerate(train_loader):

        train= Variable(images.view(-1,28*28))
        labels=Variable(labels)

        #clear grads
        optimizer.zero_grad()

        #forward prop
        outputs=model(train)

        #loss
        loss=error(outputs,labels)

        #calculate grads
        loss.backward()

        #update params
        optimizer.step()

        count+=1

        if count%50==0:
            correct=0
            total=0

            for images,labels in test_loader:
                
                test=Variable(images.view(-1,28*28))
                outputs=model(test)
                #print(outputs) 
                predicted=torch.max(outputs.data,1)[1]#max prob value 
                total+=len(labels)

                correct+=(predicted ==labels).sum()

            acc=100*correct/float(total)

            loss_list.append(loss.data)
            iter_list.append(count)
            acc_list.append(acc)

        if count %100==0:
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