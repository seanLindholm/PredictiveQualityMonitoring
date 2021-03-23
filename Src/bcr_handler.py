import numpy as np
import os
import pandas as pd
import cv2
from keyboard import * 

#path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
#file_path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\"

path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
file_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\"
failed = file_path + "failed_ext.csv"
approved = file_path + "approved_ext.csv"

failed_NoNaN = file_path + "failed_noNaN.csv"
approved_NoNaN = file_path + "approved_noNaN.csv"


def extendNoNaNData(file,ext_file):
    df = getData(file)
    print(df.columns)
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

    #extendNoNaNData(failed,failed_NoNaN)
    #extendNoNaNData(approved,approved_NoNaN)
    
    #df = removeNANrows(getData(failed),"ï»¿bcr_dir")
    #saveDF(df,failed_NoNaN)
    #df = removeNANrows(getData(approved),"ï»¿bcr_dir")
    #saveDF(df,approved_NoNaN)
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
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',na_values='')


def removeNANrows(df,column_name):
    #print(df)
    return df[df[column_name].notna()]



if __name__ == "__main__":
    main()