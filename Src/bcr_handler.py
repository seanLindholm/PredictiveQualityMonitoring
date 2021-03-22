import numpy as np
import os
import pandas as pd
import cv2
from keyboard import * 

failed = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\failed_ext.csv"
approved = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\approved_ext.csv"

failed_NoNaN = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\failed_noNaN.csv"
approved_NoNaN = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\approved_noNaN.csv"

path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
def extendNoNaNData(file,ext_file):
    df = getData(file)
    df = removeNANrows(df,'bcr_dir')
    if df.empty:
        print("Datafile is empty, aboard..")
        return
    else:
        df2 = getData(ext_file)
        df = df.append(df2,ignore_index=True,sort=False)
        df.to_csv(ext_file,encoding='latin_1',index=False)
        df = df.iloc[0:0]
        df.to_csv(file,encoding='latin_1',index=False)
    

def BcrToJpg(data_file):
    df = getData(data_file)
    counter = 1
    for folder in df['bcr_dir']:     
        n_f = openAndCloseDirectory(path + folder,counter)
        if(n_f > 3): print(f"Jpg already saved for {folder}");counter+=1;continue
        print(f"Folder number: {counter}")
        for i in range(n_f):
            time.sleep(1)
            NavigateBCR(not i)
            time.sleep(3)
            saveToJpg()
            time.sleep(1)
            closeBCRfile()
        time.sleep(1)
        closeFolder()
        counter+=1
            
    
    

def main():

    extendNoNaNData(failed,failed_NoNaN)
    extendNoNaNData(approved,approved_NoNaN)
    
    BcrToJpg(failed_NoNaN)
    BcrToJpg(approved_NoNaN)


        
    

def openAndCloseDirectory(path,counter):
    size = len(os.listdir(path))
    if (size <= 3):
        os.startfile(path)
    return size

def saveDF(df,name):
    df.to_csv(name,index=False,encoding='latin-1')



def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',encoding='latin_1',na_values='')


def removeNANrows(df,column_name):
    return df[df[column_name].notna()]



if __name__ == "__main__":
    main()