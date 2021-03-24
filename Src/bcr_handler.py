import numpy as np
import os
import pandas as pd
import cv2
from keyboard import * 
from constants import *


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
            
    
def removeDublicates():
    #This function removes dupilcates and the folders that was wrongly labelled good and bad
    df_f = getData(failed_NoNaN)
    df_a = getData(approved_NoNaN)
    folder_del = []
    df_f_del = []
    df_a_del = []
    for index,r in df_f.iterrows():
        folder = r['bcr_dir'][:-6] + "Approved"
        hit = df_a.loc[(df_a['bcr_dir'] == folder)]["bcr_dir"]
        if(not df_a.index[(df_a['bcr_dir'] == folder)].empty):
            folder_del.append(hit.tolist()[0])
            folder_del.append(r['bcr_dir'])
            df_f_del.append(index)
            df_a_del.append(hit.index.tolist()[0])
    df_f.drop(df_f_del,inplace=True)
    df_a.drop(df_f_del,inplace=True)
    saveDF(df_f,failed_NoNaN)
    saveDF(df_a,approved_NoNaN)
    for f in folder_del:
        os.system("rmdir "+path+f + " /s /q")


def main():
    #extendNoNaNData(failed,failed_NoNaN)
    #extendNoNaNData(approved,approved_NoNaN)
    removeDublicates()
    
    #df = removeNANrows(getData(failed),"ï»¿bcr_dir")
    #saveDF(df,failed_NoNaN)
    #df = removeNANrows(getData(approved),"ï»¿bcr_dir")
    #saveDF(df,approved_NoNaN)
    #BcrToJpg(failed_NoNaN)
    #BcrToJpg(approved_NoNaN)


        
    

def openAndCloseDirectory(path,counter):
    size = len(os.listdir(path))
    if (size <= 3):
        os.startfile(path)
    return size



def removeNANrows(df,column_name):
    #print(df)
    return df[df[column_name].notna()]



if __name__ == "__main__":
    main()