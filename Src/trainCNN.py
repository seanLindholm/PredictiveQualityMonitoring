from ANN import CNN,device
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
    y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)

    net = CNN(1).to(device)
    data = loadImageData("numpyData\\img_data_innerCircle_YM")
    split = int(data.shape[0]*0.8)
    data_indx = np.random.permutation(data.shape[0])
    X_train = data[data_indx[:split]][:]
    X_test = data[data_indx[split:]][:]
    y_train = y[data_indx[:split]][:]
    y_test = y[data_indx[split:]][:]
    print(X_train.shape)

    hist_loss = np.array([])
    hist_acc = np.array([])
    train_step = 48
    test_step = int(train_step/4)
    tail_train = 0;train_start = train_step
    tail_test = 0; test_start = test_step

    while True:
        net.train(X_train[tail_train:train_start][:],X_test[tail_test:test_start],y_train[tail_train:train_start][:],y_test[tail_test:test_start],epochs=100)
        hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
        hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
        tail_train+=train_step;train_start+=train_step
        tail_test+=test_step;test_start+=test_step
        if (tail_train > X_train.shape[0]):
            break

    plot(hist_loss,hist_acc)

    img = X_test[55].reshape(1,1,256,256)
    dea=net.forward(img)
    img = img[0][0]*255
    img = img.astype('uint8')
    cv2.imshow('Original',img)
    print(dea,y_test[55])
    cv2.waitKey(0)
    
       
if __name__ == "__main__":
    main(True)