from ANN import CNN,device,torch
import pandas as pd
import cv2
import numpy as np 
from constants_ import *
import matplotlib.pyplot as plt
from imgProccessing import extractMidSection as ems




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



        saveImageData("numpyData\\img_data_in_innerCircle_YM","in_inner_crop_YM.jpg",path=path)
        print("Done file YM in_inner circle")

    #The DEA score data
    df = getData(failed_DEA)
    df_a = getData(approved_DEA)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)
    y = np.append(np.zeros(df.shape[0]),np.ones(getData(approved_DEA).shape[0])).reshape(-1,1).astype('float32')
    #data = loadImageData("numpyData\\img_data_bothCircles_CA")
    #data = loadImageData("numpyData\\img_data_bothCircles_RAW")

    #data = loadImageData("numpyData\\img_data_bothCircles_YM")
    data = loadImageData("numpyData\\img_data_in_innerCircle_YM")
    #data = loadImageData("numpyData\\img_data_innerCircle_YM")
    
    
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


    big_picture = False
    net = CNN(1,big_picture=big_picture,classPrediction=True,early_stopping=True).to(device)
    torch.save(net.state_dict(), path+"YM_in_inner")

    hist_loss = np.array([])
    hist_acc = np.array([])

    
    net.train(X_train,X_test,y_train,y_test,epochs=2000)
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