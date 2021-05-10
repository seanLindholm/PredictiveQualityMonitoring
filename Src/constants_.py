import pandas as pd
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest as isoF
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
file_path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\"
data_path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\"

# path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
# file_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\"
# data_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\"
dummy_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\dummy_data\\"
failed = file_path + "failed_ext.csv"
failed_ext_norm = file_path + "failed_ext_Norm.csv"
failed_NoNaN = file_path + "failed_NoNaN.csv"
failed_DEA = file_path + "failed_withDEAScore.csv"

approved = file_path + "approved_ext.csv"
approved_ext_norm = file_path + "approved_ext_Norm.csv"
approved_NoNaN = file_path + "approved_NoNaN.csv"
approved_DEA = file_path + "approved_withDEAScore.csv"

pure_img_approved = data_path + "Poor_func_approved.csv"
pure_img_falied = data_path + "Poor_func_failed.csv" 


function_test_col_transformed = ["2/1 mM Glu/Lac [mM]","1 mM H2O2 [mM]","40/25 mM glu/lac høj O2",
                                 "Sensitivity [pA/µM]","t on 10/5 mM glu/lac [s]","Lav O2 - Høj O2"]

fcnn_data = ["time_betw_scan_min","CA","CA Humidity","CA Temperature","YM","YM Humidity","YM Temperature","CA_cav_dia","CA_void_cav","CA_cav_depth","Ca_void_mem","CA_overlap_min","CA_overlap_max","YM_void_mem"]
#fcnn_data = ["time_betw_scan_min","CA","CA Humidity","CA Temperature","YM","YM Humidity","YM Temperature","CA_cav_dia","CA_cav_depth","CA_overlap_min","CA_overlap_max"]

def saveDF(df,name):
    df.to_csv(name,index=False)

def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',na_values='')

def normalize(numpy_data):
    """
        Normalizes the datatable in the range (0-1) 
        making use of the min value and peek-to-peek range (max-min)
        and the minimum
    """
    org_dt_min =numpy_data.min(0)
    org_dt_ptp = numpy_data.ptp(0)
    numpy_data = (numpy_data - org_dt_min) / org_dt_ptp
    return numpy_data
        

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

def outlierRemoval(file,contamination='auto'):
    """
        This outlier detector makes use of the Isolated tree method, from
        the sklearn library.
    """
    df = getData(file)
    print(df.columns)

    df_2 = df[function_test_col_transformed]
    col = df_2.to_numpy()
    iso = isoF(contamination=contamination)
    yhat = iso.fit_predict(col.reshape(-1,6))
    mask = yhat != -1
    #print(mask.reshape(-1,1))
    index_del = []
    for i in range(df.shape[0]):
        if(~mask[i]): index_del.append(i)
    df = df.drop(index_del)
    saveDF(df,file)

def statisticalAnalysis(file,columns):
    #["2/1 mM Glu/Lac [mM]","1 mM H2O2 [mM]","40/25 mM glu/lac høj O2","Sensitivity [pA/µM]","t on 10/5 mM glu/lac [s]","Lav O2 - Høj O2"]
    df = getData(file)
    df = df[columns]
    data_num = df.to_numpy()
    print(f"Statistics for {file}")
    for i in range(len(columns)):
        print(f"column: {columns[i]}")
        print(f"std: {np.std(data_num[:,i])}")
        print(f"var: {np.var(data_num[:,i])}")
        print(f"mean: {np.mean(data_num[:,i])}")
        print()

def lowestComponancePCA(dt,explain_var,min_comp=1):
        """
            This function builds finds the lowest number of Principal Components
            that is equal to or above the specified explain_variance (range 0-1)
        """
        for i in range(min_comp,dt.shape[1]+1):
            pca = PCA(n_components=i)
            pca.fit(dt)
            if(sum(pca.explained_variance_ratio_) >= explain_var):
                return pca
        return pca


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


def scrampleAndSplitData(df,df_a,ImageData=False,numpy_data_name="",WithDEA = False,out_parameters = []):
    if WithDEA:
        out_parameters = []
    elif out_parameters != []:
        WithDEA = False
    if ImageData:
        data = loadImageData(numpy_data_name)
    else:
        data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    #80 % train 20% test
    split = int(df.shape[0]*0.8)
    #random indecies
    f_data_indx = np.random.permutation(df.shape[0])

    #50/50 failed and approved
    data_f = data[:df.shape[0],:][f_data_indx]
    if (WithDEA):
        y_f = df['DEA'].to_numpy().reshape(-1,1)[f_data_indx]
    elif out_parameters != []:
        y_f = normalize(df[out_parameters].to_numpy().reshape(-1,1)[f_data_indx])
    else:
        y_f = np.zeros(df.shape[0]).reshape(-1,1)

    a_data_indx = np.random.permutation(df_a.shape[0])[:df.shape[0]]
    data_a = (data[df.shape[0]:,:])[a_data_indx]
    if (WithDEA):
        y_a = (df_a['DEA'].to_numpy())[a_data_indx].reshape(-1,1)
    elif out_parameters != []:
        y_a = normalize((df_a[out_parameters].to_numpy())[a_data_indx].reshape(-1,1))
    else:
        y_a = np.ones(df.shape[0]).reshape(-1,1)

    #Build data train and test
    X_train = np.append(data_f[:split,:],data_a[:split,:],axis=0)
    X_test = np.append(data_f[split:,:],data_a[split:,:],axis=0)
    y_train = np.append(y_f[:split,:],y_a[:split,:],axis=0)
    y_test = np.append(y_f[split:,:],y_a[split:,:],axis=0)

    #Shuffle test and train
    train_shuffle = np.random.permutation(X_train.shape[0])
    test_shuffle = np.random.permutation(X_test.shape[0])
    X_train = X_train[train_shuffle].astype('float32') 
    X_test = X_test[test_shuffle].astype('float32') 
    y_train = y_train[train_shuffle].astype('float32') 
    y_test = y_test[test_shuffle].astype('float32') 

    return X_train,X_test,y_train,y_test