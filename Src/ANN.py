import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from Helper import Data_handler
from torch import nn, optim, utils
import matplotlib.pyplot as plt



#The path
data_path = "..\\Data"
eff_mixed_center_name = data_path + "\\dea_eff_centroid.csv"
mixed_transform = data_path + "\\mixed_with_clusters.csv"

# Global device check, if GPU is available use it!
if torch.cuda.is_available():  
    device = torch.device('cuda')
else:  
    device = torch.device('cpu')

class Net(nn.Module):
    def __init__(self,in_features):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.dh_classTarget = Data_handler(file_path_csv=eff_mixed_center_name)
        self.bounds = np.array([ [i-j,i+j] for i,j in self.dh_classTarget.dt])

        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)




        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=0.0001)
        #Loss
        self.loss = self.loss_function

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

        #Squeeze the output between 0-1 with sigmoid
        output = torch.sigmoid(x)
        return output

    def loss_function(self,pred,target):   
        # First create the multiplyer of 2 for each none correct class labeling
        # The class labeling is defined by std range for each class
        # This is attached in self.dh_classTarget
        indexs = target[:,2].reshape(1,-1).int()
        if(indexs.size()[0] < 2):
            target_class = self.bounds[indexs]
        else:
            index = indexs.squeeze()
            target_class = torch.mean(torch.tensor(self.bounds[indexs]),dim=0)
        p = torch.mean(pred.detach())
        loss = torch.mean(torch.abs((pred - target[:,1].reshape(-1,1))**2))
        penalty = 1 if target_class[0] <= p and p <= target_class[1] else 4
        return loss*penalty
        
    def calc_accuracy(self,pred,target):
        indexs = target[:,2].cpu().reshape(1,-1).squeeze().int()
        target_class = torch.mean(self.bounds[indexs],dim=0)
        p = torch.mean(pred.detach().cpu().numpy())
        lb = p >= target_class[:,0].reshape(-1,1)
        ub = p <= target_class[:,1].reshape(-1,1)
        good = [l_t and u_t for l_t,u_t in zip(lb,ub)]
        return np.mean(good)


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
                out = self.loss(pred,Variable(y).to(device))
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
                acc += 1-torch.mean(torch.abs(pred-Variable(y[:,1].reshape(-1,1)).to(device)))
            
            acc /= len(X_test) 
            
            self.epoch_acc.append(acc) 

            # # -- Early stopping -- #

            if abs(acc-old_acc) < threshold:
                 if max_iter == 10:
                     self.epochs = e+1
                     break
                 max_iter +=1
            else:
                 max_iter = 0
                 old_acc = acc 

            print(f"{e}/{epochs}",flush=True)
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
        
        plt.figure('Accuratcy')
        plt.plot(self.epoch_acc)
        plt.legend(["acc"])
        plt.show(block=False)
        input("Press enter to close all windows")
        plt.close('all')

def splitData(X,y,proc_train,seed = None):
    np.random.seed(seed)
    Ran_indecies = np.random.permutation(np.arange(X.shape[0]))
    train_stop = int(X.shape[0]*proc_train)
    np.random.seed(None)
    return X[Ran_indecies[:train_stop],:].float(),y[Ran_indecies[:train_stop],:].float(),X[Ran_indecies[train_stop:],:].float(),y[Ran_indecies[train_stop:],:].float()



def createBatch(X_train,y_train,X_test,y_test,batch_size=32):
    #Wraps around if dataset is not devicable with 32 (append the first diff to the end of each )
    def data_batch(X,batch_size):
        if (X.shape[0]%batch_size) != 0:
            slice_ = batch_size - (X.shape[0] % batch_size)
            X = torch.cat((X,X[:slice_,:]),dim=0)

        X_batch = []
        start_indx = 0
        end_indx = batch_size
        while (X.shape[0] >= end_indx):
            X_batch.append(X[start_indx:end_indx,:])
            start_indx += batch_size
            end_indx += batch_size
        
        return X_batch

    return data_batch(X_train,batch_size),data_batch(y_train,batch_size),data_batch(X_test,batch_size),data_batch(y_test,batch_size)


    
    
def main():

    #This will be used as part of the cost-function
    dh_eff_cent = Data_handler(file_path_csv=eff_mixed_center_name)
    
    #This is the data that needs to be trained on
    dh_data = Data_handler(file_path_csv=mixed_transform)
    
    #Split the data in in_ out_ and select the dea_eff score as label
    dh_data.splitData(3)
    X,y = torch.tensor(dh_data.dt_in),torch.tensor(dh_data.dt_out)
    #Split data in train and test with 80 % train and 20 % test
    X_train,y_train,X_test,y_test = splitData(X,y,0.8,seed=42)    
    X_train,y_train,X_test,y_test = createBatch(X_train,y_train,X_test, y_test,batch_size=1)

    my_nn = Net(in_features = 3).to(device)
    print(my_nn)

    #Construct the network with the appropiate number of input data for each sample
    print(my_nn.train(X_train,y_train,X_test,y_test,epochs=100))
    my_nn.plot()
    

if __name__ == "__main__":
    main()



