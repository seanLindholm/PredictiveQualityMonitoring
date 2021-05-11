import numpy as np
import os
import pandas as pd
import cv2
from keyboard import * 
from constants_ import *
from imgProccessing import CreateAndSaveImgs


failed_NoNaN = file_path + "failed_noNaN.csv"
approved_NoNaN = file_path + "approved_noNaN.csv"



    
def listDicts(path):
        for dict_ in os.listdir(path):
            yield dict_

def BcrToJpg():
    #df = getData(data_file)
    counter = 1
    for folder in listDicts(path):
        if(folder != "Model_ScanDATA"):
            n_f = openAndCloseDirectory(path + folder,counter)
            if(n_f > 3): print(f"Jpg already saved for {folder}");counter+=1;continue
            print(f"Folder number: {counter} - {folder}")
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
            
def cleanOutAllJpg():
    for folder in listDicts(path):
        for file_ in listDicts(path + folder):
            if ".jpg" in file_:
                os.system(f"del /f {path + folder}\\{file_}")

def removeDublicates():
    #This functi
    # on removes dupilcates and the folders that was wrongly labelled good and bad
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
    df_a.drop(df_a_del,inplace=True)
    saveDF(df_f,failed_NoNaN)
    saveDF(df_a,approved_NoNaN)
    if ("y" == input(f"Found {len(folder_del)} folders for deletion in remove dublicates, should thye be deleted? [Y/N]")):
        for f in folder_del:
            os.system("rmdir "+path+f + " /s /q")

def removeOddSizeImages():
    #This function removes any images that isn't 256x256 (for the inner cicrle)
    #Or 482x512 for both/outter circle
    #Usely happens with the scanner being a bit off center
    folder_del = []
    next_folder = False
    for folder in listDicts(path):
        for file_ in ["outter_crop_RAW","both_crop_RAW","outter_crop_CA","both_crop_CA","outter_crop_YM","both_crop_YM","inner_crop_RAW","inner_crop_CA","inner_crop_YM"]:
            p = path + folder
            #print(p+"\\"+ file_+".jpg")
            try:
                img = cv2.cvtColor(cv2.imread(p+"\\"+ file_+".jpg"), cv2.COLOR_BGR2GRAY)
                if(file_[-3] != "R"):
                    img2 = cv2.cvtColor(cv2.imread(p+"\\"+ file_+"_diff.jpg"), cv2.COLOR_BGR2GRAY)
                else:
                    img2 = img
                if ((img.shape[0] == 256 or img.shape[0] == 482) and (img2.shape[0] == 256 or img2.shape[0] == 482) ):
                    continue
                else:
                    folder_del.append(folder)
                    break
            except Exception:
                folder_del.append(folder)

    if ("y" == input(f"Found {len(folder_del)} folders for deletion in removeOddSize, should thye be deleted? [Y/N]")):
        for f in folder_del:
            os.system("rmdir \""+path+f + "\" /s /q")


def main():
    #removeDublicates()
    
    #df = removeNANrows(getData(failed),"bcr_dir")
    #saveDF(df,failed_NoNaN)
    #df = removeNANrows(getData(approved),"bcr_dir")
    #saveDF(df,approved_NoNaN)
    #cleanOutAllJpg()
    #BcrToJpg()
    #CreateAndSaveImgs(False)
    #CreateAndSaveImgs(True)

    #print("Removing odd sized imgaes")
    #removeOddSizeImages()
    
    ##List variance,mean and std-deviation of the different parameters 
    #statisticalAnalysis(failed_NoNaN,function_test_col_transformed)
    #statisticalAnalysis(approved_NoNaN,function_test_col_transformed)
    #statisticalAnalysis(failed,fcnn_data)
    #statisticalAnalysis(approved,fcnn_data)
    
    #  



        
    

def openAndCloseDirectory(path,counter,doIt=False):
    size = len(os.listdir(path))
    if (size <= 3 or doIt):
        os.startfile(path)
    return size



def removeNANrows(df,column_name):
    #print(df)
    return df[df[column_name].notna()]



if __name__ == "__main__":
    main()