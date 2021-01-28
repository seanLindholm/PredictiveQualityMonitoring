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
    def __init__(self,shape):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(shape[1], 128)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)




        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=0.001)
        #Loss
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

        #Squeeze the output between 0-1 with sigmoid
        output = torch.sigmoid(x)
        return output

    def loss_function(self,pred,target):      
        loss = torch.abs(pred - target)
        return loss
        
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

            
            loss_e /= X_train.shape[0]
            self.epoch_loss.append((loss_e))

            # -- Testing -- #
            acc = 0
            for X,y in zip(X_test,y_test):
                pred = self.forward(Variable(X).to(device))
                acc += 1-torch.abs(pred-Variable(y).to(device))
            
            acc /= X_test.shape[0] 
            
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
        plt.plot(self.epoch_loss)
        plt.plot(self.epoch_acc)
        plt.legend(["loss","acc"])

        plt.show()

def splitData(X,y,proc_train,seed = None):
    np.random.seed(seed)
    Ran_indecies = np.random.permutation(np.arange(X.shape[0]))
    train_stop = int(X.shape[0]*proc_train)
    return X[Ran_indecies[:train_stop],:].float(),y[Ran_indecies[:train_stop],:].float(),X[Ran_indecies[train_stop:],:].float(),y[Ran_indecies[train_stop:],:].float()





def main():
    #This will be used as part of the cost-function
    dh_eff_cent = Data_handler(file_path_csv=eff_mixed_center_name)
    
    #This is the data that needs to be trained on
    dh_data = Data_handler(file_path_csv=mixed_transform)
    
    #Split the data in in_ out_ and select the dea_eff score as label
    dh_data.splitData(3)
    X,y = torch.tensor(dh_data.dt_in),torch.tensor(dh_data.dt_out[:,1].reshape(-1,1))
    #Split data in train and test with 80 % train and 20 % test
    X_train,y_train,X_test, y_test = splitData(X,y,0.8,seed=1337)
   
    #Construct the network with the appropiate number of input data for each sample
    my_nn = Net(X_train.shape).to(device)

    print(my_nn)
    print(my_nn.train(X_train,y_train,X_test,y_test,epochs=100))
    my_nn.plot()
if __name__ == "__main__":
    main()



