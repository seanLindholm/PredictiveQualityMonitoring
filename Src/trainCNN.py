from ANN import CNN,device,torch
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
    net = CNN(1).to(device)
    net.train(X_train,X_train,X_test,X_test,epochs=1000)
    net.plot()
    for img,expect in zip(X_train,X_test):
        print(net.forward(img.reshape(1,1,256,256)),expect)

def main(SaveImgData=False):
    if SaveImgData:
        saveImageData("numpyData\\img_data_innerCircle_YM","inner_crop_YM.jpg",path=path)
        print("Done file ym inner circle")
        saveImageData("numpyData\\img_data_innerCircle_CA","inner_crop_CA.jpg",path=path)
        print("Done file ca inner circle")
        saveImageData("numpyData\\img_data_innerCircle_RAW","inner_crop_RAW.jpg",path=path)
        print("Done file RAW inner circle")
        
        saveImageData("numpyData\\img_data_bothCircles_YM","both_crop_YM.jpg",path=path)
        print("Done file ym outter circle")
        saveImageData("numpyData\\img_data_bothCircles_CA","both_crop_CA.jpg",path=path)
        print("Done file ca outter circle")
        saveImageData("numpyData\\img_data_bothCircles_RAW","both_crop_RAW.jpg",path=path)
        print("Done file RAW outter circle")

    #The DEA score data
    df = getData(failed_DEA)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)
    y_f = np.append(np.ones(df.shape[0]).reshape(-1,1),np.zeros(df.shape[0]).reshape(-1,1),axis=1)
    y_a = np.append(np.zeros(getData(approved_DEA).shape[0]).reshape(-1,1),np.ones(getData(approved_DEA).shape[0]).reshape(-1,1),axis=1)
    y = np.append(y_f,y_a,axis=0).astype('float32')
    #y = np.append(np.zeros(df.shape[0]),np.ones(getData(approved_DEA).shape[0])).reshape(-1,1).astype('int64')
    #data = loadImageData("numpyData\\img_data_innerCircle_YM")
    #data = loadImageData("numpyData\\img_data_bothCircles_CA")
    data = loadImageData("numpyData\\img_data_bothCircles_YM")

    big_picture = True
    net = CNN(1,big_picture=big_picture,classPrediction=True,early_stopping=False).to(device)
    torch.save(net.state_dict(), path+"YM_both")
    split = int(data.shape[0]*0.8)
    data_indx = np.random.permutation(data.shape[0])
    X_train = data[data_indx[:split]][:]
    X_test = data[data_indx[split:]][:]
    y_train = y[data_indx[:split]][:]
    y_test = y[data_indx[split:]][:]

    hist_loss = np.array([])
    hist_acc = np.array([])

  
    net.train(X_train,X_test,y_train,y_test,epochs=200)
    hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
    hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
    

    plot(hist_loss,hist_acc)

    for ind in np.random.permutation(X_test.shape[0])[:15]:
        if big_picture:
            img = X_test[ind].reshape(1,1,482,512)
        else:
            img = X_test[ind].reshape(1,1,256,256)
        dea=net.forward(img)
        img = img[0][0]*255
        img = img.astype('uint8')
        cv2.imshow('Original',img)
        print(dea,y_test[ind])
        cv2.waitKey(0)
    torch.save(net.state_dict(), path+"YM_both")
       
if __name__ == "__main__":
    main(False)