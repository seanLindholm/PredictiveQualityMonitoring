from constants_ import *
from sklearn import svm

def main():
     #The DEA score data
    df = getData(failed)
    df_a = getData(approved)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)


    for _ in range(10):
        data = normalize(df.append(df_a)[fcnn_data].to_numpy())
        X_train,X_test,y_train,y_test= scrampleAndSplitData(data,df,df_a,WithDEA = False)

        #the svm model, with linear kernel
        clf = svm.SVC()
        clf.fit(X_train,y_train.squeeze())
        pred = clf.predict(X_test)
        print(acc_clf(pred.reshape(-1,1),y_test))

def acc_clf(pred,target):
    return (pred.shape[0] - (sum(abs(pred-target)))) / pred.shape[0] * 100

if __name__ == "__main__":
    main()