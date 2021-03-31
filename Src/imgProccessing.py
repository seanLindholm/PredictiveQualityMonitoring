import cv2 
import numpy as np 
import os
import matplotlib.pyplot as plt
from constants_ import *


def listDicts(path):
        for dict_ in os.listdir(path):
            yield dict_

def findDictMatch(dict_,path):
    for d in listDicts(path):
        if d == dict_:
            return path + "\\" + dict_
    return ""


def findFilesMatch(file_,path):
    for f in listDicts(path):
        if f == file_:
            return path + "\\" + file_

    return ""


def innerCircle(img):
    return MaskCircles(img,128)

def outerCircle(img):
    mask1,crop = MaskCircles(img)
    mask2,_ = MaskCircles(img,130)
    return mask1 & ~mask2,crop

def bothCircles(img):
    return MaskCircles(img)

def MaskCircles(img,radius=256):
    # location and size of the circle
    xc, yc = detectSensor(img) 
    r = radius
    
    # size of the image
    H, W = img.shape[0],img.shape[1]

    # The tight crop around the circle
    x_margl = int(yc-r) if int(yc-r) > 0 else 0
    x_margr = int(yc+r) if int(yc+r) < H else H
    y_margl = int(xc-r) if int(xc-r) > 0 else 0
    y_margr = int(xc+r) if int(xc+r) < W else W


    crop = [x_margl,x_margr,y_margl,y_margr]
    # x and y coordinates per every pixel of the image
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    # squared distance from the center of the circle
    d2 = (x - xc)**2 + (y - yc)**2
    # mask is True inside of the circle
    return  d2 < r**2,crop
    
def find_directory(dirName, search_path):

    # Wlaking top-down from the root
    for root, dir, files in os.walk(search_path):
            if dirName in dir:
                return os.path.join(root, dirName)
    return ""

def CreateAndSaveImgs():
    list_picName = ["RAW.jpg","YM.jpg","CA.jpg"]
    list_function = [innerCircle,outerCircle,bothCircles]
    list_fileName = ["inner_crop","outter_crop","both_crop"]
    counter = 1
    for folder in listDicts(path):
        for picName in list_picName:
            pic = path+folder+"\\"+picName
            img = cv2.imread(pic)
            for func,save_name in zip(list_function,list_fileName):
                try:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mask,tight_crop = func(gray)
                    gray[~mask] = 255
                    image = gray[tight_crop[0]:tight_crop[1],tight_crop[2]:tight_crop[3]]
                    # cv2.imshow('Cropped image',image)
                    # cv2.waitKey(0)
                    cv2.imwrite(path+folder+"\\"+save_name+"_"+picName,image)
                except:
                    print(f"Problem at: {pic}")
               
        print(f"Done: {counter}")
        counter+=1

def detectSensor(img):
    gray = img
    # if(img.shape[2] > 0): 
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,
                             param1=50,param2=30,minRadius=100,maxRadius=130)
    circles = np.uint16(np.around(circles))
    
    
    for i in circles[0,:]:
        return i[0],i[1]
        # draw the outer circle
        cv2.circle(gray,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(gray,(i[0],i[1]),2,(0,0,255),3)


def extractMidSection(img):
    gray = img
    if(img.shape[2] > 0):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid_x = int(gray.shape[0]/2)
    mid_y = int(gray.shape[1]/2)

    section_x = (gray[mid_y,:])
    section_y = (gray[:,mid_x])

    print(gray.shape)
    print(section_x.shape)
    print(section_y.shape)
    

    x = np.arange(section_x.shape[0])
    y = np.arange(section_y.shape[0])

    fig, axs = plt.subplots(2,2)
    fig.suptitle('Mid sections row, column and together')
    axs[0,0].plot(x,section_x,color='blue')
    axs[0,1].plot(y,section_y,color='orange')
    axs[1,0].plot(x,section_x,color='blue')
    axs[1,0].plot(y,section_y,color='orange')
    axs[1,1].imshow(gray,cmap='gray', vmin=0, vmax=255)
    axs[1,1].plot(x,np.full(x.shape,mid_x),color='blue')
    axs[1,1].plot(np.full(y.shape,mid_y),y,color='orange')


    axs[0,0].set_title("Row mid section")
    axs[0,1].set_title("Column mid section")
    axs[1,0].set_title("plottet together")
    axs[1,1].set_title("Sensor")


    plt.show()

