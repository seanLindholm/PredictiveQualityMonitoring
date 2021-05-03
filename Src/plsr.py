import numpy as np 
from constants_ import *
from sklearn.cross_decomposition import PLSRegression as plsr
from sklearn.metrics import mean_squared_error

def mse(a1,a2):
    difference_array = np.subtract(a1, a2)
    squared_array = np.square(difference_array)
    return squared_array.mean()

def main(SaveImgData=False):
   
    #The DEA score data
    df = getData(failed_DEA)
    df_a = getData(approved_DEA)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)
    y = np.append(np.zeros(df.shape[0]),np.ones(getData(approved_DEA).shape[0])).reshape(-1,1).astype('float32')

    data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    # pca = lowestComponancePCA(data,0.95)
    # print(pca.explained_variance_ratio_.sum())
    # print(pca.explained_variance_)
    
    #This is done for debugging
    #indexs = np.arange(data.shape[0]).reshape(-1,1)
    #data = np.append(indexs,data,axis=1)
   
    # data = pca.fit_transform(data)
    
    #80 % train 20% test
    #random indecies
    f_data_indx = np.random.permutation(df.shape[0])
    #50/50 failed and approved
    data_f = data[:df.shape[0],:][f_data_indx]
    y_f = df['DEA'].to_numpy().reshape(-1,1)[f_data_indx]
    #y_f = np.zeros(df.shape[0]).reshape(-1,1)

    #Insure that a_data_indx only have as many as we have failed (50/50) set
    a_data_indx = np.random.permutation(df_a.shape[0])[:df.shape[0]]
    data_a = (data[df.shape[0]:,:])[a_data_indx]
    y_a = (df_a['DEA'].to_numpy())[a_data_indx].reshape(-1,1)
    #y_a = np.ones(df.shape[0]).reshape(-1,1)


    split = int(df.shape[0]*0.8)

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
    

    run_res = []
    x = np.arange(1,len(fcnn_data))
    for i in x:
        pls = plsr(n_components=i)
        pls.fit(X_train,(y_train*100))
        y_pred = pls.predict(X_test)
        run_res.append(mse(X_test,(y_test*100)))
    
    pls = plsr(n_components=(np.argmin(np.array(run_res))+1) )
    pls.fit(X_train,(y_train*100))
    y_pred = pls.predict(X_test)
    for yp,yt in zip(y_pred,(y_test*100)):
        print(f"prediction: {yp} vs actual: {yt}")

    plt.plot(x,run_res)
    plt.show()
        
    
if __name__ == "__main__":
    main(False)