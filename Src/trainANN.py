from ANN import FCNN,device,torch
import pandas as pd
import cv2
import numpy as np 
from constants_ import *
import matplotlib.pyplot as plt




def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',encoding='latin_1',na_values='')



def loadImageData(load_name):
    data = np.load(load_name+".npz")
    for i in data:
        return np.array(data[i])

def plot(loss,acc):
        plt.figure('Loss and accuracy')
        plt.plot(loss)
        plt.plot(acc)
      
        plt.legend(["loss","acc"])

        plt.figure('Loss')
        plt.plot(loss)
        plt.legend(["loss"])
        
        plt.figure('Accuracy')
        plt.plot(acc)
        plt.legend(["acc"])
     
        plt.show(block=False)
        input("Press enter to close all windows")
        plt.close('all')

def test():
    X_train = []
    for i in range(1,4):
        for im in ["_025_jpg.jpg","_05_jpg.jpg","_085_jpg.jpg"]:
            img = cv2.cvtColor(cv2.imread(dummy_path+"images\\"+str(i)+im), cv2.COLOR_BGR2GRAY)
            img = img.astype('float32');img /= 255; img = img.reshape(1,img.shape[0],img.shape[1])
            X_train.append(img)
    
    X_test = np.array([0.25,0.5,0.85,0.25,0.5,0.85,0.25,0.5,0.85],np.single).reshape(-1,1)
    X_train = np.array(X_train,np.single)
    net = FCNN(1).to(device)
    net.train(X_train,X_train,X_test,X_test,epochs=1000)
    net.plot()
    for img,expect in zip(X_train,X_test):
        print(net.forward(img.reshape(1,1,256,256)),expect)

def main(SaveImgData=False):
   
    #The DEA score data
    df = getData(failed)
    df_a = getData(approved)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)
    y = np.append(np.zeros(df.shape[0]),np.ones(getData(approved_DEA).shape[0])).reshape(-1,1).astype('float32')

    data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    # pca = lowestComponancePCA(data,0.95)
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_)
    
    # data = pca.fit_transform(data)
    
    #80 % train 20% test
    split = int(df.shape[0]*0.8)
    #random indecies
    f_data_indx = np.random.permutation(df.shape[0])

    #50/50 failed and approved
    data_f = data[:df.shape[0],:][f_data_indx]
    #y_f = df['DEA'].to_numpy().reshape(-1,1)[f_data_indx]
    y_f = np.zeros(df.shape[0]).reshape(-1,1)

    a_data_indx = np.random.permutation(df_a.shape[0])[:df.shape[0]]
    data_a = (data[df.shape[0]:,:])[a_data_indx]
    #y_a = (df_a['DEA'].to_numpy())[a_data_indx].reshape(-1,1)
    y_a = np.ones(df.shape[0]).reshape(-1,1)

    #Build data train and test
    X_train = np.append(data_f[:split,:],data_a[:split,:],axis=0)
    X_test = np.append(data_f[split:,:],data_f[split:,:],axis=0)
    y_train = np.append(y_f[:split,:],y_a[:split,:],axis=0)
    y_test = np.append(y_f[split:,:],y_a[split:,:],axis=0)

    #Shuffle test and train
    train_shuffle = np.random.permutation(X_train.shape[0])
    test_shuffle = np.random.permutation(X_test.shape[0])
    X_train = X_train[train_shuffle].astype('float32') 
    X_test = X_test[test_shuffle].astype('float32') 
    y_train = y_train[train_shuffle].astype('float32') 
    y_test = y_test[test_shuffle].astype('float32') 

    one_counter=0
    zero_counter=0
    
    
    net = FCNN(data.shape[1],early_stopping=False,class_prediction=True).to(device)

    hist_loss = np.array([])
    hist_acc = np.array([])

   
    net.train_(X_train,X_test,y_train,y_test,epochs=200)
    hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
    hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
    torch.save(net.state_dict(), path+"Model_ScanDATA")
    

    plot(hist_loss,hist_acc)
    corr = 0
    fail = 0
    for ind in range(X_test.shape[0]):
        class_=net.forward(X_test[ind].reshape(1,-1))
        print(torch.round(class_),y_test[ind])
        if(torch.round(class_.detach().cpu()).numpy()[0] == y_test[ind]):
            corr+=1
        else:
            fail += 1
       
    print(f"Correct: {corr}, Failed: {fail}, acc {corr/(corr+fail) * 100}")
    print()
       
if __name__ == "__main__":
    main(False)