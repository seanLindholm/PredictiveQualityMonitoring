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


def main(SaveImgData=False):
   
    #The DEA score data
    df_f = getData(failed_NoNaN)
    df_a = getData(approved_NoNaN)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)

    # pca = lowestComponancePCA(data,0.95)
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_)
    
    # data = pca.fit_transform(data)

    
    one_counter=0
    zero_counter=0
    
    
    classPred = True
    tests = 10
    param_hist_loss = [i for i in range(len(function_test_col))]
    param_hist_acc = [i for i in range(len(function_test_col))]
    #for param,i in zip(function_test_col,range(len(function_test_col))):
    max_acc = 0
    lowest_RMSE = 1000
    acc_avg = 0
    for _ in range(tests):
        X_train,X_test,y_train,y_test,_,_= scrampleAndSplitData(df_f,df_a)#,numpy_data_name="numpyData\\img_data_split_YM")#,out_parameters=[param])


        
        net = FCNN(X_train.shape[1],early_stopping=True,class_prediction=classPred).to(device)

        hist_loss = np.array([])
        hist_acc = np.array([])

        acc = net.train_(X_train,X_test,y_train,y_test,epochs=200)
        acc_avg += acc
        if (classPred):
            
            if (acc > max_acc): 
                max_acc = acc 
                # param_hist_loss[i] = net.epoch_loss
                # param_hist_acc[i] = net.epoch_acc
        else:
            
            if (acc < lowest_RMSE): 
                max_acc = acc
                lowest_RMSE = acc
                # param_hist_loss[i] = net.epoch_loss
                # param_hist_acc[i] = net.epoch_acc
        
        hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
        hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
        net.plot()


    #for param,i in zip(function_test_col,range(len(function_test_col))): 
        #print(f"Acc and lost for {param} - with best accuracy")
        #plot(param_hist_loss[i],param_hist_acc[i])

    
    print(f"After {tests} runs of 200 epochs we get a max accuracy of {max_acc*100} with an average of {(acc_avg/tests)*100}")
    
    #test other methods
    # plsr - data reduction and func estimation - DEA
    # add features from images - mid section


    net = FCNN(X_train.shape[1],early_stopping=False,class_prediction=classPred).to(device)
    net.load_state_dict(torch.load(data_path+"Model_ScanDATA"))
    net.eval()
    corr = 0
    fail = 0
    tp = 0; fp = 0; tn = 0; fn = 0
    for ind in range(X_test.shape[0]):
        class_=net.forward(X_test[ind].reshape(1,-1))
        if(classPred):
            print(f"pred: {torch.round(class_).cpu().detach().numpy()[0]} test: {y_test[ind]}")
            if(torch.round(class_.detach().cpu()).numpy()[0] == y_test[ind]):
                corr+=1
                if (y_test[ind]): tp +=1 
                else: tn += 1
            else:
                fail += 1
                if (y_test[ind]): fn +=1 
                else: fp += 1
        
        else:
            print(f"pred: {class_.cpu().detach().numpy()[0]} test: {y_test[ind]}, RMSError: {net.calcRecClassPred(class_.cpu().detach(),torch.tensor(y_test[ind]))}")
    if (classPred):
        print(f"Correct: {corr}, Failed: {fail}, acc {corr/(corr+fail) * 100}")
        print(f"True positive: {tp}\nTrue negative {tn}\nFalse positive: {fp}\nFalse negative: {fn}")

    print()
       
if __name__ == "__main__":
    main(False)