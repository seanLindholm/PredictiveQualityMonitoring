from ANN import AE,device
from imgProccessing import find_directory as fd
import pandas as pd
import cv2
import numpy as np 
import matplotlib.pyplot as plt
failed = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\failed_noNaN.csv"
approved = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\approved_noNaN.csv"




def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',encoding='latin_1',na_values='')

def saveImageData(save_name,file_to_save,path=""):
    print("Saving image data");couter = 1
    l = [];p = [failed,approved]
    for f in p:
        df = getData(f);couter = 1
        for folder in df['bcr_dir']:  
            p = path + "\\" + folder
            try:
                img = cv2.cvtColor(cv2.imread(p+"\\"+ file_to_save), cv2.COLOR_BGR2GRAY)
                img = img.astype('float32');img /= 255;img = img.reshape(1,img.shape[0],img.shape[1])

                l.append(img);couter+=1
                if(couter == 25):
                    np.savez(save_name+".npz",l)
                    couter = 1
            except:
                pass
    np.savez(save_name+".npz",l)

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
        path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files"
        saveImageData("numpyData\\img_data_innerCircle_YM","inner_crop_YM.jpg",path=path)
        print("Done file ym 1")
        saveImageData("numpyData\\img_data_bothCircles_YM","both_crop_YM.jpg",path=path)
        print("Done file ym 2")

        saveImageData("numpyData\\img_data_innerCircle_CA","inner_crop_CA.jpg",path=path)
        print("Done file ca 1")
        saveImageData("numpyData\\img_data_bothCircles_CA","both_crop_CA.jpg",path=path)
        print("Done file ca 2")
  


    net = AE(1).to(device)
    data = loadImageData("numpyData\\img_data_innerCircle_CA")
    split = int(data.shape[0]*0.8)
    data_indx = np.random.permutation(data.shape[0])
    train = data[data_indx[:split]][:]
    test = data[data_indx[split:]][:]

    hist_loss = np.array([])
    hist_acc = np.array([])
    train_step = 16
    test_step = int(train_step/4)
    tail_train = 0;train_start = train_step
    tail_test = 0; test_start = test_step
    while True:
        net.train(train[tail_train:train_start][:],test[tail_test:test_start],epochs=10)
        hist_loss = np.append(hist_loss,net.epoch_loss,axis=0)
        hist_acc = np.append(hist_acc,net.epoch_acc,axis=0)
        tail_train+=train_step;train_start+=train_step
        tail_test+=test_step;test_start+=test_step
        if (tail_train > train.shape[0]):
            break

    plot(hist_loss,hist_acc)

    img = test[55].reshape(1,1,256,256)
    rec,_=net.forward(img)
    img = img[0][0]*255
    img = img.astype('uint8')
    cv2.imshow('Original',img)
    rec = rec.cpu().detach().numpy()
    rec = rec[0][0]*255
    rec = rec.astype('uint8')
    cv2.imshow('Reconstruction',rec)

    cv2.waitKey(0)
    
       
if __name__ == "__main__":
    main()