from ANN import MM,device,torch
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
    
        
        saveImageData("numpyData\\img_data_bothCircles_YM","both_crop_YM.jpg",path=path)
        print("Done file ym both circle")
        saveImageData("numpyData\\img_data_bothCircles_CA","both_crop_CA.jpg",path=path)
        print("Done file ca both circle")
        saveImageData("numpyData\\img_data_bothCircles_RAW","both_crop_RAW.jpg",path=path)
        print("Done file RAW both circle")

        saveImageData("numpyData\\img_data_outterCircles_YM","outter_crop_YM.jpg",path=path)
        print("Done file ym outter circle")
        saveImageData("numpyData\\img_data_outterCircles_CA","outter_crop_CA.jpg",path=path)
        print("Done file ca outter circle")
        saveImageData("numpyData\\img_data_outterCircles_RAW","outter_crop_RAW.jpg",path=path)
        print("Done file RAW outter circle")


        saveImageData("numpyData\\img_data_outterCircles_YM","in_inner_crop_YM_diff_strip.jpg",path=path)
        print("Done file ym split")
        saveImageData("numpyData\\img_data_outterCircles_CA","in_inner_crop_CA_diff_strip.jpg",path=path)
        print("Done file ym split")
       
        saveImageData("numpyData\\img_data_innerCircle_YM_diff","inner_crop_YM_diff.jpg",path=path)
        print("Done file ym inner circle")
        saveImageData("numpyData\\img_data_innerCircle_CA_diff","inner_crop_CA_diff.jpg",path=path)
        print("Done file ca inner circle")
        saveImageData("numpyData\\img_data_innerCircle_RAW_diff","inner_crop_RAW_diff.jpg",path=path)
        print("Done file RAW inner circle")

        
        saveImageData("numpyData\\img_data_bothCircles_YM_diff","both_crop_YM_diff.jpg",path=path)
        print("Done file ym both circle")
        saveImageData("numpyData\\img_data_bothCircles_CA_diff","both_crop_CA_diff.jpg",path=path)
        print("Done file ca both circle")
        saveImageData("numpyData\\img_data_bothCircles_RAW_diff","both_crop_RAW_diff.jpg",path=path)
        print("Done file RAW both circle")

        saveImageData("numpyData\\img_data_outterCircles_YM_diff","outter_crop_YM_diff.jpg",path=path)
        print("Done file ym outter circle")
        saveImageData("numpyData\\img_data_outterCircles_CA_diff","outter_crop_CA_diff.jpg",path=path)
        print("Done file ca outter circle")
        saveImageData("numpyData\\img_data_outterCircles_RAW_diff","outter_crop_RAW_diff.jpg",path=path)
        print("Done file RAW outter circle")

    img_data = "numpyData\\img_plot"

    #saveImageData(img_data,"outter_crop_YM_diff.jpg",path=path)
    #print("Done file " + img_data)
 

    df = getData(failed_NoNaN)
    df_a = getData(approved_NoNaN)
   

    one_counter=0
    zero_counter=0
    
    tests = 10
    param_hist_loss = [i for i in range(len(function_test_col))]
    param_hist_acc = [i for i in range(len(function_test_col))]
    #for param,i in zip(function_test_col,range(len(function_test_col))):
    max_acc = 0
    lowest_RMSE = 1000
    acc_avg = 0
    for _ in range(tests):
        X_train,X_test,y_train,y_test,X_train2,X_test2= scrampleAndSplitData(df,df_a,ImageData=True,plusMore=True,numpy_data_name=img_data)#,out_parameters=["40/25 mM glu/lac høj O2"])
        net = MM(1,len(fcnn_data),classPrediction=True,early_stopping=False).to(device)
        hist_loss = np.array([])
        hist_acc = np.array([])
        acc = net.train_(X_train,X_test,X_train2,X_test2,y_train,y_test,epochs=250)
        acc_avg += acc
        if (acc > max_acc): 
            max_acc = acc 
            # param_hist_loss[i] = net.epoch_loss
            # param_hist_acc[i] = net.epoch_acc
       
        
        hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
        hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
        #net.plot()


    # for param,i in zip(function_test_col,range(len(function_test_col))): 
    #     print(f"Acc and lost for {param} - with best accuracy")
    #     plot(param_hist_loss[i],param_hist_acc[i])

    
    print(f"After {tests} runs of 125 epochs we get a max accuracy of {max_acc*100} with an average of {(acc_avg/tests)*100}")
    


    X_train,X_test,y_train,y_test,X_train2,X_test2= scrampleAndSplitData(df,df_a,ImageData=True,plusMore=True,numpy_data_name=img_data)
    net = MM(1,len(fcnn_data),classPrediction=True,early_stopping=False).to(device)
    net.load_state_dict(torch.load(data_path+"Model_ScanDATA"))
    net.eval()
    corr = 0
    fail = 0
    tp = 0; fp = 0; tn = 0; fn = 0
    for ind in range(X_test.shape[0]):
        class_=net.forward(X_test[ind].reshape(1,1,482,512),X_test2[ind].reshape(1,-1))
        print(f"pred: {torch.round(class_).cpu().detach().numpy()[0]} test: {y_test[ind]}")
        if(torch.round(class_.detach().cpu()).numpy()[0] == y_test[ind]):
                corr+=1
                if (y_test[ind]): tp +=1 
                else: tn += 1
        else:
            fail += 1
            if (y_test[ind]): fn +=1 
            else: fp += 1
    
    print(f"Correct: {corr}, Failed: {fail}, acc {corr/(corr+fail) * 100}")
    print(f"True positive: {tp}\nTrue negative {tn}\nFalse positive: {fp}\nFalse negative: {fn}")

    print()

       
if __name__ == "__main__":
    main(False)