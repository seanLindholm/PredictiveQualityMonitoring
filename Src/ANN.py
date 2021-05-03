import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
from Helper import Data_handler, progressBar
from torch import nn, optim, utils
import matplotlib.pyplot as plt




# Global device check, if GPU is available use it!
if torch.cuda.is_available():  
    device = torch.device('cuda')
else:  
    device = torch.device('cpu')

class FCNN(nn.Module):
    def __init__(self,in_features,class_prediction=False,early_stopping=True):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.class_prediction = class_prediction
        self.early_stopping = early_stopping

        super(FCNN, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features, 256)
        # Second fully connected layer that outputs our 10 labels
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,64 )

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)



        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=10e-5)
        #Loss
        #self.loss = self.loss_function
        if self.class_prediction:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()

    def forward(self,x):
        if not torch.is_tensor(x):
            x = Variable(torch.tensor(x)).to(device)
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
            output = torch.sigmoid(x)
        else:
            output = F.relu(x)
        return output

           
    def calcAccClassPred(self,pred,test):
        count = 0
        max_ = pred.shape[0]
        for guess,exp in zip(pred,test):
            if(torch.round(guess)==exp):
                count += 1
        return count

    def train_(self,X_train,X_test,y_train,y_test,epochs=10):
        X_train = torch.tensor(X_train) if not torch.is_tensor(X_train) else X_train
        X_test = torch.tensor(X_test) if not torch.is_tensor(X_test) else X_test
        y_train = torch.tensor(y_train) if not torch.is_tensor(y_train) else y_train
        y_test = torch.tensor(y_test) if not torch.is_tensor(y_test) else y_test
        

        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0

        for e in range(epochs): 
            loss_train = 0
            acc_test = 0

            for X_t,y_t in batchSplit(X_train,y_train,batch_size=4):
                # --- Trainig ----- #
                self.train()
                self.optimizer.zero_grad()
                pred = self(Variable(X_t).to(device))
                out = self.loss(pred,Variable(y_t).to(device))
                out.backward()

                self.optimizer.step()
                loss_train += out.data.cpu().numpy()

            for X_tst,y_tst in batchSplit(X_test,y_test,batch_size=4):
                # -- Testing -- #
                self.eval()

                pred = self(Variable(X_tst).to(device))
                #Loss is with same input picture after decoding (Reconstruction loss)
                out = self.loss(pred,Variable(y_tst).to(device))

                acc_test += self.calcAccClassPred(pred.data.cpu(),y_tst)
                #print(f"prediction: {pred}, expected: {y_tst}, accuracy: {100-(abs(pred.data.cpu()-y_tst))}")

            acc_test /= X_test.shape[0]
            loss_train /= X_train.shape[0]

            self.epoch_loss.append(loss_train)
            self.epoch_acc.append(acc_test) 

            # # -- Early stopping -- #
            if self.early_stopping:
                if abs(acc_test-old_acc) < threshold:
                    if max_iter == 30:
                        e = epochs
                        break
                    max_iter +=1
                else:
                    max_iter = 1
                    old_acc = acc_test 

            print(f"\r{e+1}/{epochs}",end='\r')
            # --- return acc after trainig --- #
        return acc_test


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



class AE(nn.Module):
    """
        Expects a picture of any dimention as input
    """
    def __init__(self,in_channels,early_stopping=True):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.early_stopping = early_stopping

        super(AE, self).__init__()
        #Encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        #self.linear = nn.Linear(32768,8192)
        #self.t_linear = nn.Linear(8192,32768)

        #Decoder
        self.t_conv3 = nn.ConvTranspose2d(2, 8, 2,stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2,stride=2)
        self.t_conv1 = nn.ConvTranspose2d(16, in_channels, 1)



        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=10e-4)

        #Loss        
        self.loss = nn.BCELoss()

    def forward(self,x):
        if not torch.is_tensor(x):
            x = Variable(torch.tensor(x)).to(device)
        samples= x.shape[0]
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        x = F.relu(self.conv3(x))

        #print(x.shape)
        code = x.flatten(start_dim=1)
        #code = self.linear(x)
        #x = self.t_linear(code)
        #x = x.reshape(samples,2,128,128)
        #print(code.shape)
        #input("")
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv2(x))
        x = torch.sigmoid(self.t_conv1(x))

        return x,code

    def train(self,X_train,X_test,epochs=10):
        X_train = torch.tensor(X_train) if not torch.is_tensor(X_train) else X_train
        X_test = torch.tensor(X_test) if not torch.is_tensor(X_test) else X_test


        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0
        for e in range(epochs): 
            
            # --- Trainig ----- #
            loss_e = 0
    
            self.optimizer.zero_grad()
            pred,_ = self(Variable(X_train).to(device))
            #Loss is with same input picture after decoding (Reconstruction loss)
            out = self.loss(pred,Variable(X_train).to(device))
            out.backward()

            self.optimizer.step()
            loss_e = out.data.cpu().numpy()

            self.epoch_loss.append((loss_e))
           
            # -- Testing -- #
            loss_train = 0

            pred,_ = self(Variable(X_test).to(device))
            #Loss is with same input picture after decoding (Reconstruction loss)
            out = self.loss(pred,Variable(X_test).to(device))
            loss_train = out.data.cpu().numpy()
            
            

            self.epoch_acc.append(loss_train) 

            # # -- Early stopping -- #
            if self.early_stopping:
                if abs(loss_train-old_acc) < threshold:
                    if max_iter == 150:
                        e = epochs
                        break
                    max_iter +=1
                else:
                    max_iter = 1
                    old_acc = loss_train 

            print(f"\r{e+1}/{epochs}",end='\r')
        # --- return acc after trainig --- #

        return loss_train


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


class CNN(nn.Module):
    """
        Expects a picture of any dimention as input
    """
    def __init__(self,in_channels,early_stopping=True,big_picture=False,classPrediction=False):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.early_stopping = early_stopping
        self.classPrediction = classPrediction
        super(CNN, self).__init__()
        #Encoder
        self.conv2 = nn.Conv2d(64, 16, 3, padding=1,stride=2)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1,stride=2)
        


        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(p=0.3)
        
        if big_picture:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1,stride=2) 
            self.linear1 = nn.Linear(7680,512)
            self.linear2 = nn.Linear(512,64)
            self.linear3 = nn.Linear(64,1)
          


        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)  
            self.linear1 = nn.Linear(8192,512)
            self.linear2 = nn.Linear(512,64) 
            self.linear3 = nn.Linear(64,1)
            
      
        nn.init.xavier_uniform_(self.conv1.weight) 

        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=10e-4)

        #Loss        
        if (self.classPrediction):
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.MSELoss()

        

    def forward(self,x):
        if not torch.is_tensor(x):
            x = Variable(torch.tensor(x)).to(device)
        samples= x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        #x = F.relu(self.conv4(x))
        #x = self.pool(x)
        #x = F.relu(self.conv5(x))
        
        x = x.flatten(start_dim=1)
        x=F.relu(self.linear1(x))
        x=F.relu(self.linear2(x))
        if(self.classPrediction):
            x = torch.sigmoid(self.linear3(x))
        else:
            x=F.relu(self.linear3(x))


        return x

    def train(self,X_train,X_test,y_train,y_test,epochs=10):
        X_train = torch.tensor(X_train) if not torch.is_tensor(X_train) else X_train
        X_test = torch.tensor(X_test) if not torch.is_tensor(X_test) else X_test
        y_train = torch.tensor(y_train) if not torch.is_tensor(y_train) else y_train
        y_test = torch.tensor(y_test) if not torch.is_tensor(y_test) else y_test
        

        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0

        for e in range(epochs): 
            loss_train = 0
            acc_test = 0

            for X_t,y_t in batchSplit(X_train,y_train):
                # --- Trainig ----- #
                #torch.autograd.set_detect_anomaly(True)
                y_t=y_t.reshape(-1,1)
                if(not self.classPrediction):
                    y_t = (y_t*100)


            
                # --- Trainig ----- #
        
                self.optimizer.zero_grad()
                #print(Variable(X_t).shape)
                pred = self(Variable(X_t).to(device))
                #if(self.classPrediction):
                #    pred = torch.argmax(pred).to(device)
                #Loss is with same input picture after decoding (Reconstruction loss)
                out = self.loss(pred,Variable(y_t).to(device))
                out.backward()

                self.optimizer.step()
                loss_train += out.data.cpu().numpy()

            for X_tst,y_tst in batchSplit(X_train,y_train):
                y_tst = y_tst.reshape(-1,1)
                if(not self.classPrediction):
                    y_tst = (y_tst*100)

                # -- Testing -- #

                pred = self(Variable(X_tst).to(device))
                #Loss is with same input picture after decoding (Reconstruction loss)
                out = self.loss(pred,Variable(y_tst).to(device))
                if not self.classPrediction:
                    acc_test += torch.mean((abs(pred.data.cpu()-y_tst)))
                else:
                    acc_test += (abs(pred.data.cpu()-y_tst)).sum()
                #print(f"prediction: {pred}, expected: {y_tst}, accuracy: {100-(abs(pred.data.cpu()-y_tst))}")


                
                


            acc_test /= X_test.shape[0]
            loss_train /= X_train.shape[0]

            self.epoch_loss.append(loss_train)
            self.epoch_acc.append(acc_test) 

            # # -- Early stopping -- #
            if self.early_stopping:
                if abs(acc_test-old_acc) < threshold:
                    if max_iter == 10:
                        e = epochs
                        break
                    max_iter +=1
                else:
                    max_iter = 1
                    old_acc = acc_test 

            print(f"\r{e+1}/{epochs}",end='\r')
            # --- return acc after trainig --- #

        return loss_train


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

class RNN(nn.Module):
    """
        Expects a picture of any dimention as input
    """
    def __init__(self,early_stopping=True):
        #this is used in the loss function to give a higher penelty if it also misses the correct class label
        self.early_stopping = early_stopping

        super(RNN, self).__init__()
        #Encoder
        self.lstm_x = nn.LSTM(512,64,2)
        self.lstm_y = nn.LSTM(481,64,2)

        self.linear = nn.Linear(128,64)
        self.linear_1 = nn.Linear(64,1)
        self.hx = 0
        self.hy = 0


        nn.init.xavier_uniform_(self.lstm_x.weight) 
        nn.init.xavier_uniform_(self.lstm_y.weight) 

        #The optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=0.0005)

        #Loss        
        self.loss = nn.MSELoss()

    def forward(self,in_,hid_x,hid_y):
        if not torch.is_tensor(in_):
            in_ = Variable(torch.tensor(in_)).to(device)
       
        
        out_1, hidx = self.lstm_x(in_[:,:,:512],hid_x)
        out_2, hidy = self.lstm_y(in_[:,:,512:-1],hid_y)
        fin_out = F.relu(self.linear(torch.cat((out_1, out_2), dim=2)))
        fin_out = F.relu(self.linear_1(fin_out)).reshape(-1,1)
        return fin_out,hidx,hidy

    def train(self,X_train,X_test,y_train,y_test,epochs=10):
        X_train = torch.tensor(X_train) if not torch.is_tensor(X_train) else X_train
        X_test = torch.tensor(X_test) if not torch.is_tensor(X_test) else X_test
        y_train = torch.tensor(y_train) if not torch.is_tensor(y_train) else y_train
        y_test = torch.tensor(y_test) if not torch.is_tensor(y_test) else y_test
        

        self.epoch_loss = []
        self.epoch_acc = []
        old_acc = 0
        threshold = 0.01
        max_iter = 0
        hidx = (torch.randn(2,1,64).to(device),torch.randn(2,1,64).to(device))
        hidy = (torch.randn(2,1,64).to(device),torch.randn(2,1,64).to(device))

        for e in range(epochs): 
            loss_train = 0
            acc_test = 0

            for X_t,y_t,X_tst,y_tst in zip(X_train,y_train,X_test,y_test):
                # --- Trainig ----- #
                #torch.autograd.set_detect_anomaly(True)
                X_t = X_t.reshape(1,1,-1)
                y_t = (y_t*100).reshape(1,-1)
                X_tst=X_tst.reshape(1,1,-1)
                y_tst = (y_tst*100).reshape(1,-1)


                hidx = tuple([each.data for each in hidx])
                hidy = tuple([each.data for each in hidy])
                self.hx = hidx
                self.hy = hidy
                self.optimizer.zero_grad()
                pred,hidx,hidy = self(Variable(X_t).to(device),hidx,hidy)
                #Loss is with same input picture after decoding (Reconstruction loss)
                out = self.loss(pred,Variable(y_t).to(device))
                out.backward(retain_graph=True)

                self.optimizer.step()
                

                loss_train += out.data.cpu().numpy()

            
                # -- Testing -- #
                pred,_,_ = self(Variable(X_tst).to(device),hidx,hidy)
                #Loss is with same input picture after decoding (Reconstruction loss)
                out = self.loss(pred,Variable(y_tst).to(device))
               # print(f"prediction: {pred}, expected: {y_tst}, accuracy: {100-(abs(pred.data.cpu()-y_tst))}")
                acc_test += 100-(abs(pred.data.cpu()-y_tst))
                
                
                


            acc_test /= X_test.shape[0]
            loss_train /= X_train.shape[0]

            self.epoch_loss.append(loss_train)
            self.epoch_acc.append(acc_test) 

            # # -- Early stopping -- #
            if self.early_stopping:
                if abs(acc_test-old_acc) < threshold:
                    if max_iter == 50:
                        e = epochs
                        break
                    max_iter +=1
                else:
                    max_iter = 1
                    old_acc = acc_test 

            print(f"\r{e+1}/{epochs}",end='\r')
            # --- return acc after trainig --- #

        return acc_test


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

def batchSplit(X,y,batch_size=8):
    #Wraps around if dataset is not devicable with 32 (append the first diff to the end of each )
    if(X.shape[0] != y.shape[0]): raise Exception(f"X data rows {X.shape} isn't equal to {y.shape} rows")  
    start = 0
    next_ = batch_size
    while next_ <= X.shape[0]:
        yield X[start:next_],y[start:next_]
        start+=batch_size; next_ += batch_size 



    
    



