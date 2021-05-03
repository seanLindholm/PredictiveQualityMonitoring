from ANN import RNN,device,torch
import pandas as pd
import cv2
import numpy as np 
from constants_ import *
import matplotlib.pyplot as plt

def convertImg(img):
    mid_x = int(img.shape[0]/2)
    mid_y = int(img.shape[1]/2)

    section_x = (img[mid_y,:])
    section_y = (img[:,mid_x])

    
    return np.append(section_x,section_y).reshape(1,-1)
    

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

#   data = loadImageData("numpyData\\img_data_innerCircle_YM")
#    data = loadImageData("numpyData\\img_data_bothCircles_CA")
    data = loadImageData("numpyData\\img_data_bothCircles_YM")

    net = RNN().to(device)
    torch.save(net.state_dict(), path+"LSTM_YM")
    split = int(data.shape[0]*0.8)
    data_indx = np.random.permutation(data.shape[0])
    X_train_img = data[data_indx[:split]][:]
    X_test_img = data[data_indx[split:]][:]
    y_train = y[data_indx[:split]][:]
    y_test = y[data_indx[split:]][:]

    #Convert the pictures into mid sections
    conv = []
    for p in X_train_img:
        conv.append(convertImg(p[0]))
    X_train = np.array(conv)

    conv = []
    for p in X_test_img:
        conv.append(convertImg(p[0]))
    X_test = np.array(conv)
    

    hist_loss = np.array([])
    hist_acc = np.array([])
    train_step = 32
    
    test_step = int(train_step/4)
    tail_train = 0;train_start = train_step
    tail_test = 0; test_start = test_step

    
    net.train(X_train,X_test,y_train,y_test,epochs=1000)
    hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
    hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
    tail_train+=train_step;train_start+=train_step
    tail_test+=test_step;test_start+=test_step
    

    plot(hist_loss,hist_acc)

    # Signal processing - 
    # Fue transform 
    # welsh transform

    for ind in np.random.permutation(X_test_img.shape[0])[:5]:
        dea,_,_=net.forward(X_test[ind].reshape(1,1,-1),net.hx,net.hy)
        img = X_test_img[ind].reshape(1,1,482,512)
        img = img[0][0]*255
        img = img.astype('uint8')
        cv2.imshow('Original',img)
        print(dea,y_test[ind])
        cv2.waitKey(0)
    torch.save(net.state_dict(), path+"LSTM_YM")

       
if __name__ == "__main__":
    main(False)