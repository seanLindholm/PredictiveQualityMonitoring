import cv2 
import numpy as np 
import os

path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"


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
    xc, yc, r = img.shape[1]/2, img.shape[0]/2, radius
    
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
    

def main():
    list_picName = ["RAW.jpg","YM.jpg","CA.jpg"]
    list_function = [innerCircle,outerCircle,bothCircles]
    list_fileName = ["inner_crop","outter_crop","both_crop"]
    for folder in listDicts(path):
        for picName in list_picName:
            pic = path+folder+"\\"+picName
            img = cv2.imread(pic)
            for func,save_name in zip(list_function,list_fileName):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mask,tight_crop = func(gray)
                gray[~mask] = 255
                image = gray[tight_crop[0]:tight_crop[1],tight_crop[2]:tight_crop[3]]
                cv2.imwrite(path+folder+"\\"+save_name+"_"+picName,image)
        print("Done: " + folder)


def find_directory(dirName, search_path):

    # Wlaking top-down from the root
    for root, dir, files in os.walk(search_path):
            if dirName in dir:
                return os.path.join(root, dirName)
    return ""

if __name__ == "__main__":
    main()