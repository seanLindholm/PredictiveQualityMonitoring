from ANN import FCNN,device,torch
import pandas as pd
import cv2
import numpy as np 
from constants_ import *
import matplotlib.pyplot as plt


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

def scrampleAndSplitData(data,df,df_a):
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
    X_test = np.append(data_f[split:,:],data_a[split:,:],axis=0)
    y_train = np.append(y_f[:split,:],y_a[:split,:],axis=0)
    y_test = np.append(y_f[split:,:],y_a[split:,:],axis=0)

    #Shuffle test and train
    train_shuffle = np.random.permutation(X_train.shape[0])
    test_shuffle = np.random.permutation(X_test.shape[0])
    X_train = X_train[train_shuffle].astype('float32') 
    X_test = X_test[test_shuffle].astype('float32') 
    y_train = y_train[train_shuffle].astype('float32') 
    y_test = y_test[test_shuffle].astype('float32') 

    return X_train,X_test,y_train,y_test

def main(SaveImgData=False):
   
    #The DEA score data
    df = getData(failed)
    df_a = getData(approved)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)

    data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    # pca = lowestComponancePCA(data,0.95)
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_)
    
    # data = pca.fit_transform(data)

    X_train,X_test,y_train,y_test= scrampleAndSplitData(data,df,df_a)
    one_counter=0
    zero_counter=0
    max_acc = 0
    acc_avg = 0
    for _ in range(10):
        X_train,X_test,y_train,y_test= scrampleAndSplitData(data,df,df_a)
        net = FCNN(data.shape[1],early_stopping=False,class_prediction=True).to(device)

        hist_loss = np.array([])
        hist_acc = np.array([])

        acc = net.train_(X_train,X_test,y_train,y_test,epochs=200)
        acc_avg += acc
        if (acc > max_acc): 
            max_acc = acc
            torch.save(net.state_dict(), path+"Model_ScanDATA")
        hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
        hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
    print(f"After 10 runs of 200 epochs we get a max accuracy of {max_acc*100} with an average of {(acc_avg/10)*100}")
    
    #test other methods
    # plsr - data reduction and func estimation - DEA
    # add features from images - mid section


    net = FCNN(data.shape[1],early_stopping=False,class_prediction=True).to(device)
    net.load_state_dict(torch.load(path+"Model_ScanDATA"))
    net.eval()
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