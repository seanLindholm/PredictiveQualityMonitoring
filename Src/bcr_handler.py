import numpy as np
import os
import pandas as pd
import cv2
failed = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\failed_noNaN.csv"
approved = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\approved_noNaN.csv"



def main():
    df = getData(failed)
    counter = 1
    for path in df['bcr_dict']:        
        openAndCloseDirectory(path,counter)
        counter+=1


def openAndCloseDirectory(path,counter):
    os.startfile(path)
    input("Path number: " + str(counter))

def saveDF(df,name):
    df.to_csv(name,index=False,encoding='latin-1')



def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',encoding='latin_1',na_values='')


def removeNANrows(df,column_name):
    return df[df[column_name].notna()]



if __name__ == "__main__":
    main()