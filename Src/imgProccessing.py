import cv2 
import numpy as np 
import os
import matplotlib.pyplot as plt
from constants_ import *
from scipy.signal import find_peaks


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

def inner_innerCircle(img):
    mask1,crop = MaskCircles(img,128)
    mask2,_ = MaskCircles(img,80)
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
def createDiffernece():
    counter = 1
    for folder in listDicts(path):
        if (folder != "Model_ScanDATA"):
            print(f"Folder number {counter} - {folder}")
            pic_raw = path+folder+"\\RAW.jpg"
            pic_raw = cv2.imread(path+folder+"\\RAW.jpg")
            pic_ca = abs(cv2.imread(path+folder+"\\CA.jpg") - pic_raw)
            pic_ym = abs(abs((cv2.imread(path+folder+"\\YM.jpg") - pic_raw)) - pic_ca)
            cv2.imwrite(path+folder+"\\CA_diff.jpg",pic_ca)
            cv2.imwrite(path+folder+"\\YM_diff.jpg",pic_ym)
        counter += 1
        
               

def CreateAndSaveImgs():
    createDiffernece()
    list_picName = ["RAW.jpg","YM.jpg","CA.jpg"]
    list_function = [innerCircle,outerCircle,bothCircles,inner_innerCircle]
    list_fileName = ["inner_crop","outter_crop","both_crop","in_inner_crop"]
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


def extractMidSection(gray,RET=False):
    # Signal processing - 
    # Fue transform 
    # welch method
    
    # Make an average reference section for x and y for the three scans
    # one feature could be number of peaks from the subtracted images (removes systematic noise) 
    mid_x = int(gray.shape[0]/2)
    mid_y = int(gray.shape[1]/2)

    section_x = (gray[mid_y,:])
    section_y = (gray[:,mid_x])
    if RET: 
        return section_x,section_y
    
    

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

def getApprovedInFolder():
     # get all the folders in path, and remove the -failed end and replace with 0
    list_ = [ l.replace("-Failed", "-1") for l in os.listdir(path) ]
    del_item = []
    # now for all folders with a 1 as last car append til list
    for l in list_:
        if l[-1] == "1":
            del_item.append(l)
    
    #And remove them for org list
    for l in del_item:
        list_.remove(l)
    return list_

def getFailedInFolder():
     # get all the folders in path, and remove the -failed end and replace with 0
    list_ = [ l.replace("-Approved", "-1") for l in os.listdir(path) ]
    del_item = []
    # now for all folders with a 1 as last car append til list
    for l in list_:
        if l[-1] == "1":
            del_item.append(l)
    
    #And remove them for org list
    for l in del_item:
        list_.remove(l)
    return list_

def generateProfile():
    '''
        This function will generatre profile images for RAW, CA and YM in both X and Y direction (so 6 in total)
        It will use the whole sensor image, but with the background cropped out to generate these images.
        It only uses the approved images to generate the refference.
        These are saved in a numpy file .npz for RAW, CA and YM 
    '''
    # get all the folders in path, and remove the -failed end and replace with 0
    list_ = [ l.replace("-Failed", "-1") for l in os.listdir(path) ]
    del_item = []
   
    for img_ in ["both_crop_RAW","both_crop_CA","both_crop_YM"]:
        x_section = np.array([[]])
        y_section = np.array([[]])
        for l in getApprovedInFolder():
            img = cv2.imread(f"{path}{l}\\{img_}.jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x,y = extractMidSection(gray,RET=True)
            if(x_section.shape[1] == 0 ): 
                x_section = x.reshape(1,-1)
            else:
                x_section = np.append(x_section,x.reshape(1,-1),axis=0)

            if(y_section.shape[1] == 0 ): 
                y_section = y.reshape(1,-1)
            else:
                y_section = np.append(y_section,y.reshape(1,-1),axis=0)
        np.savez(data_path+img_+"_x_.npz",x_section.mean(axis=0))
        np.savez(data_path+img_+"_y_.npz",y_section.mean(axis=0))

def getNumberOfPeaksThreshYM(df,threshold=35):
    load_name = ["both_crop_RAW","both_crop_CA","both_crop_YM"]
    gray = []
    img = cv2.imread(path + img_test + "\\" + name + ".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    data_x = np.load(data_path+"both_crop_YM_x_.npz")
    for i in data_x:
            profile_x = data_x[i]
        
    x_test,y_test = extractMidSection(gray,RET=True)
    x_test = abs(x_test.astype('float') - profile_x)
    x_test[x_test<35] = 0
    y_test[y_test<35] = 0
    x_peaks, _ = find_peaks(x_test)
    y_peaks, _ = find_peaks(y_test)


def getNumberOfPeaksThresh(threshold=35):
    list_app = getApprovedInFolder()
    list_fail = getFailedInFolder()
    load_name = ["both_crop_RAW","both_crop_CA","both_crop_YM"]
    gray = []
    for name in load_name:
        peaks_x = []
        peaks_y = []
        max_x = []
        max_y = []
        # for the approved ones
        acc_peaks_x = 0
        acc_peaks_y = 0
        acc_max_x = 0
        acc_max_y = 0
        for img_test in list_app:
            img = cv2.imread(path + img_test + "\\" + name + ".jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            x_test = []; y_test = []
            x_section = []; y_section = []
            data_x = np.load(data_path+name+"_x_.npz")
            data_y = np.load(data_path+name+"_y_.npz")
            
            for i in data_x:
                x_section = data_x[i]

            for i in data_y:
                y_section = data_y[i]
            
            x_x = np.arange(x_section.shape[0])
            x_y = np.arange(y_section.shape[0])

        
            x_test,y_test = extractMidSection(gray,RET=True)
            x_test = abs(x_test.astype('float') - x_section)
            y_test = abs(y_test.astype('float') - y_section)
            x_test[x_test<35] = 0
            y_test[y_test<35] = 0
            x_peaks, _ = find_peaks(x_test)
            y_peaks, _ = find_peaks(y_test)
            peaks_x.append(x_peaks)
            peaks_y.append(y_peaks)
            acc_peaks_x += len(x_peaks)
            acc_peaks_y += len(y_peaks)
            if x_peaks.size > 0:
                max_x.append(np.max(x_peaks))
                acc_max_x += np.max(x_peaks)
            if y_peaks.size > 0:
                max_y.append(np.max(y_peaks))
                acc_max_y += np.max(y_peaks)
        
        print(f"Average number of peaks for x: {acc_peaks_x/len(list_app):.2f} and y: {acc_peaks_y/len(list_app):.2f}, std x: {np.std(np.array(x_peaks)):.2f}, std y: {np.std(np.array(y_peaks)):.2f} for Approved - {name}")
        print(f"Average number of max peak x: {acc_max_x/len(list_app):.2f} and max peak y: {acc_max_y/len(list_app):.2f}, std x: {np.std(np.array(max_x)):.2f}, std y: {np.std(np.array(max_y)):.2f} for Approved - {name}")


        # for the failed ones
        acc_peaks_x = 0
        acc_peaks_y = 0
        acc_max_x = 0
        acc_max_y = 0
        peaks_x = []
        peaks_y = []
        max_x = []
        max_y = []
        for img_test in list_fail:
            img = cv2.imread(path + img_test + "\\" + name + ".jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            x_test = []; y_test = []
            x_section = []; y_section = []
            data_x = np.load(data_path+name+"_x_.npz")
            data_y = np.load(data_path+name+"_y_.npz")
            
            for i in data_x:
                x_section = data_x[i]

            for i in data_y:
                y_section = data_y[i]
            
            x_x = np.arange(x_section.shape[0])
            x_y = np.arange(y_section.shape[0])

        
            x_test,y_test = extractMidSection(gray,RET=True)
            x_test = abs(x_test.astype('float') - x_section)
            y_test = abs(y_test.astype('float') - y_section)
            x_test[x_test<35] = 0
            y_test[y_test<35] = 0
            x_peaks, _ = find_peaks(x_test)
            y_peaks, _ = find_peaks(y_test)
            peaks_x.append(x_peaks)
            peaks_y.append(y_peaks)
            acc_peaks_x += len(x_peaks)
            acc_peaks_y += len(y_peaks)
            if x_peaks.size > 0:
                max_x.append(np.max(x_peaks))
                acc_max_x += np.max(x_peaks)
            if y_peaks.size > 0:
                max_y.append(np.max(y_peaks))
                acc_max_y += np.max(y_peaks)

        
        print(f"Average number of peaks for x: {acc_peaks_x/len(list_fail):.2f} and y: {acc_peaks_y/len(list_fail):.2f}, std x: {np.std(np.array(x_peaks)):.2f}, std y: {np.std(np.array(y_peaks)):.2f} for Failed - {name}")
        print(f"Average number of max peak x: {acc_max_x/len(list_fail):.2f} and max peak y: {acc_max_y/len(list_fail):.2f}, std x: {np.std(np.array(max_x)):.2f}, std y: {np.std(np.array(max_y)):.2f} for Failed - {name}")

        print()




def plotProfile(img_test = None):
    load_name = ["both_crop_RAW","both_crop_CA","both_crop_YM"]
    gray = []
    for name in load_name:
        if img_test is not None:
            img = cv2.imread(img_test + name + ".jpg")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        x_test = []; y_test = []
        x_section = []; y_section = []
        data_x = np.load(data_path+name+"_x_.npz")
        data_y = np.load(data_path+name+"_y_.npz")
        
        for i in data_x:
            x_section = data_x[i]

        for i in data_y:
            y_section = data_y[i]
        
        x_x = np.arange(x_section.shape[0])
        x_y = np.arange(y_section.shape[0])

        if img_test is not None:
            x_test,y_test = extractMidSection(gray,RET=True)
            x_test = abs(x_test.astype('float') - x_section)
            y_test = abs(y_test.astype('float') - y_section)
            x_test[x_test<35] = 0
            y_test[y_test<35] = 0
            x_peaks, _ = find_peaks(x_test)
            y_peaks, _ = find_peaks(y_test)
         


        
        if img_test is None:
            fig, axs = plt.subplots(2)
            fig.suptitle(f'The profile for {name}')
            axs[0].plot(x_x,x_section,color='blue')
            axs[1].plot(x_y,y_section,color='orange')
        else:
            fig, axs = plt.subplots(2,2)
            fig.suptitle(f'The profile for {name} with a test image frequency subtracated')
            axs[0,0].plot(x_x,x_section,color='blue')
            axs[0,1].plot(x_y,y_section,color='orange')
            axs[1,0].plot(x_x,x_test,color='red')
            axs[1,0].plot(x_peaks,x_test[x_peaks],'X')

            axs[1,1].plot(x_y,y_test,color='green')
            axs[1,1].plot(y_peaks,y_test[y_peaks],'X')
            print(f"number of peaks for x: {len(x_peaks)}, and number of peaks for y: {len(y_peaks)}, from image {img_test} at {name}")
    print()

def invertGrayscale():
    df = getData(failed_NoNaN)
    for i in range(154,180):
        img = cv2.imread(path+df['bcr_dir'][i] + "\\in_inner_crop_YM.jpg")
        gray = 255-cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_pix  = np.max(gray)
        gray[gray<(max_pix-15)] = 0
        kernel = np.ones((5,5),np.uint8)
        gray = cv2.erode(gray,kernel)
        cv2.imshow('image',gray)
        cv2.imshow('image_org',img)
        cv2.waitKey(0)



# CreateAndSaveImgs()
# #generateProfile()
# #getNumberOfPeaksThresh()
# plotProfile()
# plotProfile(img_test=path+"932-029-R28424-N003-A5-Approved\\")
# plt.show()

# img = cv2.imread(path+"932-029-R28411-N001-A5-Failed\\both_crop_RAW.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# img = cv2.imread(path+"932-029-R28411-N001-A5-Failed\\both_crop_CA.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# img = cv2.imread(path+"932-029-R28411-N001-A5-Failed\\both_crop_YM.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# plt.show()
# img = cv2.imread(path+"932-029-R28424-N003-A5-Approved\\both_crop_RAW.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# img = cv2.imread(path+"932-029-R28424-N003-A5-Approved\\both_crop_CA.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# img = cv2.imread(path+"932-029-R28424-N003-A5-Approved\\both_crop_YM.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# extractMidSection(gray)
# plt.show()