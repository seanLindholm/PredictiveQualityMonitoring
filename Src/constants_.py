import pandas as pd
import cv2
import numpy as np
#path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
#file_path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\"

path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
file_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\"
data_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\"
failed = file_path + "failed_ext.csv"
failed_NoNaN = file_path + "failed_NoNaN.csv"
failed_DEA = file_path + "failed_withDEAScore.csv"

approved = file_path + "approved_ext.csv"
approved_NoNaN = file_path + "approved_NoNaN.csv"
approved_DEA = file_path + "approved_withDEAScore.csv"



function_test_col_transformed = ["Tid efter start [timer]","2/1 mM Glu/Lac [mM]","1 mM H2O2 [mM]","40/25 mM glu/lac høj O2",
                                 "Sensitivity [pA/µM]","t on 10/5 mM glu/lac [s]","Lav O2 - Høj O2"]


def saveDF(df,name):
    df.to_csv(name,index=False)

def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',na_values='')


def saveImageData(save_name,file_to_save,path=""):
    print("Saving image data");couter = 1
    l = [];p = [failed_NoNaN,approved_NoNaN]
    for f in p:
        df = getData(f);counter = 1
        for folder in df['bcr_dir']:  
            p = path + "\\" + folder
            try:
                img = cv2.cvtColor(cv2.imread(p+"\\"+ file_to_save), cv2.COLOR_BGR2GRAY)
                if(img.shape[0] == 482 or img.shape[0] == 256):
                    img = img.astype('float32');img /= 255;img = img.reshape(1,img.shape[0],img.shape[1])        
                    l.append(img);counter+=1
                if(counter == 25):
                    np.savez(save_name+".npz",l)
                    counter = 1
            except:
                pass
    np.savez(save_name+".npz",l)
