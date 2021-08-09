from constants_ import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
def main():
    #The DEA score data
    df = getData(failed)
    df_a = getData(approved)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)

    data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    R = abs(np.corrcoef(data.T))
    plt.matshow(R)
    plt.xticks(np.arange(data.shape[1]),fcnn_data,rotation=90)
    plt.yticks(np.arange(data.shape[1]),fcnn_data)

    plt.colorbar()
    plt.show()
    X_train,X_test,y_train,y_test= scrampleAndSplitData(data,df,df_a,WithDEA = False)

    X_bc = np.append(X_train,X_test,axis=0)
    y_bc = np.append(y_train,y_test,axis=0)


   

    df = getData(failed_DEA)
    df_a = getData(approved_DEA)
    #y = np.array(df.append(getData(approved_DEA))['DEA'],np.single).reshape(-1,1)

    data = normalize(df.append(df_a)[fcnn_data].to_numpy())
    
    X_train,X_test,y_train,y_test= scrampleAndSplitData(data,df,df_a,WithDEA = True)

    X_dea = np.append(X_train,X_test,axis=0)
    y_dea = np.append(y_train,y_test,axis=0)


    # The regression test with fittin to each functiontest parameter (one at a time)
    
    # The regression test (with the DEA scores)
    # LinearRegressionTest(X_dea,y_dea)
    # print()
    # DecisionTreeRegressorTest(X_dea,y_dea)
    # print()
    # RandomForestRegressorTest(X_dea,y_dea)
    # print()

    # # The binary classificationstests
    # LogisticRegressionTest(X_bc,y_bc)
    # print()
    # DecisionTreeClassifierTest(X_bc,y_bc)
    # print()
    # RandomForestClassifierTest(X_bc,y_bc)
    # print()


def RandomForestClassifierTest(X,y):
    # define the model
    model = RandomForestClassifier()
    # fit the model
    model.fit(X, y.ravel())
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(fcnn_data, importance)
    plt.show()

def RandomForestRegressorTest(X,y):
    # define the model
    model = RandomForestRegressor()
    # fit the model
    model.fit(X, y.ravel())
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(fcnn_data, importance)
    plt.show()

def DecisionTreeClassifierTest(X,y):
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(fcnn_data, importance)
    plt.show()

def DecisionTreeRegressorTest(X,y):
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    # plot feature importance
    plt.bar(fcnn_data, importance)
    plt.show()

def LinearRegressionTest(X,y):
    
    model = LinearRegression().fit(X,y.squeeze())
    # get importance
    importance = model.coef_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    plt.bar(fcnn_data, importance)
    plt.show()

def LogisticRegressionTest(X,y):
    model = LogisticRegression().fit(X,y)
    # get importance
    importance = model.coef_[0]
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
    plt.bar(fcnn_data, importance)
    plt.show()



if __name__ == "__main__":
    main()

