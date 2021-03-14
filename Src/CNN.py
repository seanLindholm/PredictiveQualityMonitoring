import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from Helper import progressBar
from torch import nn, optim, utils
import matplotlib.pyplot as plt

# Global device check, if GPU is available use it!
if torch.cuda.is_available():  
    device = torch.device('cuda')
else:  
    device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self,in_features,dh_classTarget,class_prediction=False,early_stopping=True):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.dh_classTarget = dh_classTarget
        self.bounds = np.array([ [i-j,i+j] for i,j in (self.dh_classTarget.dt[:,:2])])
        self.class_prediction = class_prediction
        self.early_stopping = early_stopping

        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, 32)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128,64 )

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        if not self.class_prediction:
            self.fc6 = nn.Linear(16, 1)
        else:
            self.fc6 = nn.Linear(16 ,5)



        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=10e-5)
        #Loss
        #self.loss = self.loss_function
        if self.class_prediction:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self,x):
        #Pass through first layer
        x = self.fc1(x)
        #use ReLU as activation
        x = F.relu(x)
        #Pass through second layer
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        #Squeeze the output between 0-1 with sigmoid
        if self.class_prediction:
            output = F.softmax(x)
        else:
            output = F.relu(x)
        return output

    def calc_acc(self,pred,target):
        if not self.class_prediction:
            indexs = target[:,2].reshape(1,-1).cpu().int()
            if(indexs.size()[1] < 2):
                target_class = torch.tensor(self.bounds[indexs]).to(device)
            else:
                index = indexs.squeeze()
                target_class = torch.tensor(self.bounds[indexs]).to(device).squeeze()
            p = pred.detach()
            if indexs.size()[1] > 2:
                return torch.sum(torch.tensor([ (bound[0] <= x and x <= bound[1]) for x,bound in zip(p,target_class)])).to(device)
            else:
                return torch.sum(torch.tensor([ (target_class[0] <= p and p <= target_class[1])] )).to(device)
        else:
            return -1
        
           

    def loss_function(self,pred,target):   
        # First create the multiplyer of 2 for each none correct class labeling
        # The class labeling is defined by std range for each class
        # This is attached in self.dh_classTarget
        if not self.target_param:
            indexs = target[:,2].reshape(1,-1).cpu().int()
            if(indexs.size()[1] < 2):
                target_class = torch.tensor(self.bounds[indexs]).to(device)
            else:
                index = indexs.squeeze()
                target_class = torch.tensor(self.bounds[indexs]).to(device).squeeze()
            p = pred.detach()
            if indexs.size()[1] > 2:
                p_sum = torch.sum(torch.tensor([ not (bound[0] <= x and x <= bound[1]) for x,bound in zip(p,target_class)])).to(device)
            else:
                p_sum = torch.tensor( not (target_class[0] <= p and p <= target_class[1] )).to(device)
            loss = torch.mean((pred - target[:,1].reshape(-1,1)).pow(2))
            penalty = p_sum if p_sum > 0 else torch.tensor(1).to(device)
            return loss*penalty
        else:
            return torch.mean((pred - target[:,4:]).pow(2))
        


    def train(self,X_train,y_train,X_test,y_test,epochs=10):
        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0
        for e in range(epochs): 
            
            # --- Trainig ----- #
            loss_e = 0
    
            for X,y in zip(X_train,y_train):
                self.optimizer.zero_grad()
                pred = self(Variable(X).to(device))
                if self.class_prediction:
                    out = self.loss(pred,Variable(y[:,2].long()).to(device))
                else:
                    out = self.loss(pred,Variable(y[:,1]).reshape(-1,1).to(device))

                out.backward()
                self.optimizer.step()
                loss_e += out.data.cpu().numpy()

            
            loss_e /= len(X_train)
            self.epoch_loss.append((loss_e))
           
            # -- Testing -- #
            acc = 0
            for X,y in zip(X_test,y_test):
                pred = self.forward(Variable(X).to(device))
                # y[:,1] contains the dea - effcienty scores the rest is used in the loss function
                #acc += self.calc_accuracy(pred,y)
                max_score = torch.tensor(1)
                if not self.class_prediction:
                    target_train = y
                    max_score = torch.tensor(y.shape[0]).float().to(device)
                    acc += (self.calc_acc(pred,target_train).float()/max_score)
                else:
                    target_train = y[:,2].to(device)
                    pred = torch.argmax(pred,dim=1).to(device)
                    acc += torch.divide(torch.sum(torch.eq(pred.to(device),target_train).to(device)).float(),y.shape[0])

            len(X_test)/max_score
            acc /= len(X_test)

            self.epoch_acc.append(acc) 

            # # -- Early stopping -- #
            if self.early_stopping:
                if abs(acc-old_acc) < threshold:
                    if max_iter == 150:
                        e = epochs
                        break
                    max_iter +=1
                else:
                    max_iter = 1
                    old_acc = acc 

            print(f"\r{e+1}/{epochs}",end='\r')
        # --- return acc after trainig --- #

        return acc


    def plot(self):
        plt.figure('Loss and accuracy')
        plt.plot(self.epoch_loss)
        plt.plot(self.epoch_acc)
      
        plt.legend(["loss","acc"])

        plt.figure('Loss')
        plt.plot(self.epoch_loss)
        plt.legend(["loss"])
        
        plt.figure('Accuracy')
        plt.plot(self.epoch_acc)
        plt.legend(["acc"])
     
        plt.show(block=False)
        input("Press enter to close all windows")
        plt.close('all')