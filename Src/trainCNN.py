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



    saveImageData("numpyData\\img_data_split_YM","in_inner_crop_YM_diff_strip.jpg",path=path)
    print("Done file YM in_inner circle")


    df = getData(failed_DEA)
    df_a = getData(approved_DEA)
    X_train,X_test,y_train,y_test= scrampleAndSplitData(df,df_a,ImageData=True,numpy_data_name="numpyData\\img_data_split_YM")#,out_parameters=["40/25 mM glu/lac høj O2"])
    


    big_picture = False
    split = True
    net = CNN(1,big_picture=big_picture,classPrediction=True,early_stopping=False,split=split).to(device)
    torch.save(net.state_dict(), data_path+"YM_in_inner")

    hist_loss = np.array([])
    hist_acc = np.array([])

    
    net.train_(X_train,X_test,y_train,y_test,epochs=1000)
    hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
    hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
    

    plot(hist_loss,hist_acc)

    for ind in np.random.permutation(X_test.shape[0])[:15]:
        if split:
            img = X_test[ind].reshape(1,1,32,360)
        elif big_picture:
            img = X_test[ind].reshape(1,1,482,512)
        else:
            img = X_test[ind].reshape(1,1,256,256)
        dea=net.forward(img)
        img = img[0][0]*255
        img = img.astype('uint8')
        cv2.imshow('Original',img)
        print(torch.round(dea),y_test[ind])
        cv2.waitKey(0)
    torch.save(net.state_dict(), data_path+"YM_both")
       
if __name__ == "__main__":
    main(False)