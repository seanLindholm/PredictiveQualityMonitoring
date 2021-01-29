from Helper import Data_handler, prettyprint as pp, progressBar
from Kmeans import *
from ANN import *
#The path to the data used
data_path = "..\\Data"
approved_file_name = data_path + "\\Approved_w_glu.csv"
failed_file_name = data_path + "\\Failed_w_glu.csv"
approved_file_Transform_name = data_path + "\\Approved_w_glu_Transform.csv"
failed_file_Transform_name = data_path + "\\Failed_w_glu_Transform.csv"
eff_failed_name = data_path + "\\eff_failed.csv"
eff_approved_name = data_path + "\\eff_approved.csv"
eff_mixed_name = data_path + "\\eff_mixed.csv"
# These two paths are created when buildDataForann has run
eff_mixed_center_name_5 = data_path + "\\dea_eff_centroid5.csv"
eff_mixed_center_name_4 = data_path + "\\dea_eff_centroid4.csv"
mixed_transform_5 = data_path + "\\mixed_with_clusters5.csv"
mixed_transform_4 = data_path + "\\mixed_with_clusters4.csv"


def buildMixedData():
    # Build mixed class from approved and failed data
    dh_approved = Data_handler(approved_file_Transform_name)
    dh_failed = Data_handler(failed_file_Transform_name)
    dh_approved.dt[:,3] = 1
    dh_failed.dt[:,3] = 0

    #Frist remove all outliers
    dh_approved.outlierRemoval(contamination=0.01)
    dh_failed.outlierRemoval(contamination=0.01)
    #Now make sure that there is equal amount of approved/failed arrays
    Ran_app_sel_trans = np.random.permutation(np.arange(dh_approved.dt.shape[0]))[:dh_failed.dt.shape[0]]
    dh_approved.dt = dh_approved.dt[Ran_app_sel_trans,:]

    #Construct the mixed clas from the two data_handlers
    dh_mixed = Data_handler.from2DH(dh_approved,dh_failed)
    dh_mixed.saveToCsv('Mixed_transform_withClass_unNorm')
    dh_mixed.removeColumns(['Class'])
    dh_mixed.saveToCsv('Mixed_transform_noClass_unNorm')

    dh_mixed.normalized()
    dh_mixed.saveToCsv('Mixed_transform_noClass_normalized')
    dh_mixed.restoreSavedData()
    dh_mixed.saveToCsv('Mixed_transform_WithClass_normalized')

def buildDataForAnn():
    dh_mixed = Data_handler(file_path_csv="..\\Data\\Mixed_transform_withClass_normalized.csv")
    dh_mixed.removeColumns(['Class'])

    dh_mixed.splitData(3)
    dh_mixed_eff = Data_handler(file_path_csv=eff_mixed_name)
    labels = printOutput([dh_mixed,dh_mixed_eff],start_cluster=4)

    class_eff5 = Data_handler.dataAndHeader(labels[1][2],["cluster efficency - 5 clusters"])
    class_norm5 = Data_handler.dataAndHeader(labels[0][2],["cluster normalized - 5 clusters"])

    dh_mixed.restoreSavedData()

    dh_normMixed = Data_handler(file_path_csv="..\\Data\\Mixed_transform_noClass_normalized.csv")
    dh_paramTransform_norm = Data_handler.dataAndHeader(dh_normMixed.dt[:,3:],['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    dh_unnormMixed = Data_handler(file_path_csv="..\\Data\\Mixed_transform_withClass_unNorm.csv")
    dh_unnormMixed.append(dh_mixed_eff,axis=1)
    dh_unnormMixed.append(class_eff5,axis=1)
    dh_unnormMixed.append(class_norm5,axis=1)
    dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    dh_unnormMixed.deleteSavedData()
    dh_unnormMixed.append(dh_paramTransform_norm,axis=1)
    dh_unnormMixed.saveToCsv('mixed_with_clusters5')
    

    #Now find std for the different clusters of eff
    mask_0 = np.where(dh_unnormMixed.dt[:,5] == 0)
    mask_1 = np.where(dh_unnormMixed.dt[:,5] == 1)
    mask_2 = np.where(dh_unnormMixed.dt[:,5] == 2)
    mask_3 = np.where(dh_unnormMixed.dt[:,5] == 3)
    mask_4 = np.where(dh_unnormMixed.dt[:,5] == 4)

    std = np.array([[np.std(dh_unnormMixed.dt[mask_0,4])],
                   [np.std(dh_unnormMixed.dt[mask_1,4])],
                   [np.std(dh_unnormMixed.dt[mask_2,4])],
                   [np.std(dh_unnormMixed.dt[mask_3,4])],
                   [np.std(dh_unnormMixed.dt[mask_4,4])]])

    dea_eff_centroid = Data_handler.dataAndHeader(np.append(labels[1][3],std,axis=1),["Centers - 5 cluster","std"])
    dea_eff_centroid.append(Data_handler.dataAndHeader(labels[0][3],["Normcenter 1,Normcenter 2,Normcenter 3,Normcenter 4,Normcenter 5,Normcenter 6"]),axis=1)
    dea_eff_centroid.saveToCsv('dea_eff_centroid5')

    class_eff4 = Data_handler.dataAndHeader(labels[1][0],["cluster efficency - 4 clusters"])
    class_norm4 = Data_handler.dataAndHeader(labels[0][0],["cluster normalized - 4 clusters"])

    dh_unnormMixed = Data_handler(file_path_csv="..\\Data\\Mixed_transform_withClass_unNorm.csv")
    dh_unnormMixed.append(dh_mixed_eff,axis=1)
    dh_unnormMixed.append(class_eff4,axis=1)
    dh_unnormMixed.append(class_norm4,axis=1)

    dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    dh_unnormMixed.saveToCsv('mixed_with_clusters4')
    

    #Now find std for the different clusters of eff
    mask_0 = np.where(dh_unnormMixed.dt[:,5] == 0)
    mask_1 = np.where(dh_unnormMixed.dt[:,5] == 1)
    mask_2 = np.where(dh_unnormMixed.dt[:,5] == 2)
    mask_3 = np.where(dh_unnormMixed.dt[:,5] == 3)


    std = np.array([[np.std(dh_unnormMixed.dt[mask_0,4])],
                   [np.std(dh_unnormMixed.dt[mask_1,4])],
                   [np.std(dh_unnormMixed.dt[mask_2,4])],
                   [np.std(dh_unnormMixed.dt[mask_3,4])]])


    dea_eff_centroid = Data_handler.dataAndHeader(np.append(labels[1][1],std,axis=1),["Centers - 4 cluster","std"])
    dea_eff_centroid.append(Data_handler.dataAndHeader(labels[0][1],["Normcenter 1,Normcenter 2,Normcenter 3,Normcenter 4,Normcenter 5,Normcenter 6"]),axis=1)
    dea_eff_centroid.saveToCsv('dea_eff_centroid4')




def test():
    a = np.array([[0,1,0,2,3]]).squeeze()
    b = np.array([[0.01,8],[0.02,3],[0.03,1],[0.04,4],[0.05,7]])

    x = b[a]
    for a in x:
        print(a)
    c = np.mean(np.array([ [j-i,j+i] for i,j in x]),axis=0)
    p = np.mean(np.array([[7],[4],[7],[2],[4]]))
    print(p)
    print(c)
    print(c[0] <= p and p <= c[1])
    pass

def runANN(split,seed,num_runs=1,batch_size = 8,epochs = 200,eff_center=eff_mixed_center_name_5,data=mixed_transform_5,target_param=False):
    #This will be used as part of the cost-function
    dh_eff_cent = Data_handler(file_path_csv=eff_center)
    
    #This is the data that needs to be trained on
    dh_data = Data_handler(file_path_csv=data)
    
    #Split the data in in_ out_ and select the dea_eff score as label
    dh_data.splitData(3)
    X,y = torch.tensor(dh_data.dt_in),torch.tensor(dh_data.dt_out)
    acc = 0
    acc_best = -1
    best_nn = None
    for i in range(num_runs):
        progressBar(i,num_runs)

        #Split data in train and test with 80 % train and 20 % test
        X_train,y_train,X_test,y_test = splitData(X,y,proc_train=split,seed=seed)    
        X_train,y_train,X_test,y_test = createBatch(X_train,y_train,X_test, y_test,batch_size=batch_size)
        nn = Net(in_features = 3,dh_classTarget=dh_eff_cent,target_param=target_param).to(device)


        #Construct the network with the appropiate number of input data for each sample
        acc_tmp = nn.train(X_train,y_train,X_test,y_test,epochs=epochs)
        if (acc_tmp > acc_best):
            acc_best = acc_tmp
            best_nn = nn 
        acc += acc_tmp
        progressBar(i+1,num_runs)
    
    print(f"Done traning with {num_runs} runs. The average accuracy is scored as {acc/num_runs}.")
    print("\nThe plot from the last iteration")
    nn.plot()
    return best_nn

def analyseClusterDistribution():
    dh_unnormMixed = Data_handler(file_path_csv=mixed_transform_5)
    dh_unnormMixed_4 = Data_handler(file_path_csv=mixed_transform_4)

    dh_classTarget = Data_handler(file_path_csv=eff_mixed_center_name_5)
    dh_classTarget_4 = Data_handler(file_path_csv=eff_mixed_center_name_4)

    

    #Now find std for the different clusters of eff
    mask_0 = np.where(dh_unnormMixed.dt[:,5] == 0)
    mask_1 = np.where(dh_unnormMixed.dt[:,5] == 1)
    mask_2 = np.where(dh_unnormMixed.dt[:,5] == 2)
    mask_3 = np.where(dh_unnormMixed.dt[:,5] == 3)
    mask_4 = np.where(dh_unnormMixed.dt[:,5] == 4)

    mask_0_norm = np.where(dh_unnormMixed.dt[:,6] == 0)
    mask_1_norm = np.where(dh_unnormMixed.dt[:,6] == 1)
    mask_2_norm = np.where(dh_unnormMixed.dt[:,6] == 2)
    mask_3_norm = np.where(dh_unnormMixed.dt[:,6] == 3)
    mask_4_norm = np.where(dh_unnormMixed.dt[:,6] == 4)

    
    print("Class distribtuion with 5 clusters on DEA - efficency\n")
    print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed.dt[mask_0,3].squeeze())}, centroid: {dh_classTarget.dt[0,0]:.3f}")
    print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed.dt[mask_1,3].squeeze())}, centroid: {dh_classTarget.dt[1,0]:.3f}")
    print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed.dt[mask_2,3].squeeze())}, centroid: {dh_classTarget.dt[2,0]:.3f}")
    print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed.dt[mask_3,3].squeeze())}, centroid: {dh_classTarget.dt[3,0]:.3f}")
    print(f"Class distribution in cluster 4: {Counter(dh_unnormMixed.dt[mask_4,3].squeeze())}, centroid: {dh_classTarget.dt[4,0]:.3f}\n")
    
    print("Class distribtuion with 5 clusters on normalized distance parameters\n")
    print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed.dt[mask_0_norm,3].squeeze())}, centroid: {dh_classTarget.dt[0,2:]}")
    print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed.dt[mask_1_norm,3].squeeze())}, centroid: {dh_classTarget.dt[1,2:]}")
    print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed.dt[mask_2_norm,3].squeeze())}, centroid: {dh_classTarget.dt[2,2:]}")
    print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed.dt[mask_3_norm,3].squeeze())}, centroid: {dh_classTarget.dt[3,2:]}")
    print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed.dt[mask_4_norm,3].squeeze())}, centroid: {dh_classTarget.dt[4,2:]}\n")


    #Now find std for the different clusters of eff
    mask_0 = np.where(dh_unnormMixed_4.dt[:,5] == 0)
    mask_1 = np.where(dh_unnormMixed_4.dt[:,5] == 1)
    mask_2 = np.where(dh_unnormMixed_4.dt[:,5] == 2)
    mask_3 = np.where(dh_unnormMixed_4.dt[:,5] == 3)
  
    mask_0_norm = np.where(dh_unnormMixed_4.dt[:,6] == 0)
    mask_1_norm = np.where(dh_unnormMixed_4.dt[:,6] == 1)
    mask_2_norm = np.where(dh_unnormMixed_4.dt[:,6] == 2)
    mask_3_norm = np.where(dh_unnormMixed_4.dt[:,6] == 3)

    
    print("Class distribtuion with 4 clusters on DEA - efficency\n")
    print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed_4.dt[mask_0,3].squeeze())}, centroid: {dh_classTarget_4.dt[0,0]:.3f}")
    print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed_4.dt[mask_1,3].squeeze())}, centroid: {dh_classTarget_4.dt[1,0]:.3f}")
    print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed_4.dt[mask_2,3].squeeze())}, centroid: {dh_classTarget_4.dt[2,0]:.3f}")
    print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed_4.dt[mask_3,3].squeeze())}, centroid: {dh_classTarget_4.dt[3,0]:.3f}\n")

    print("Class distribtuion with 4 clusters on normalized distance parameters\n")
    print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed_4.dt[mask_0_norm,3].squeeze())}, centroid: {dh_classTarget_4.dt[0,2:]}")
    print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed_4.dt[mask_1_norm,3].squeeze())}, centroid: {dh_classTarget_4.dt[1,2:]}")
    print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed_4.dt[mask_2_norm,3].squeeze())}, centroid: {dh_classTarget_4.dt[2,2:]}")
    print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed_4.dt[mask_3_norm,3].squeeze())}, centroid: {dh_classTarget_4.dt[3,2:]}\n")

def testBestNN(nn,data=mixed_transform_5,target=eff_mixed_center_name_5,target_param=False,ran_num_sample=10):
    dh = Data_handler(file_path_csv=data)
    dh_t = Data_handler(file_path_csv=target)

    Ran_indecies = np.random.permutation(np.arange(dh.dt.shape[0]))
    test_data = dh.dt[Ran_indecies[:ran_num_sample]]

    for i in range (ran_num_sample):
        t = torch.FloatTensor([np.array(test_data[i,:3])]).cuda()
        print(t)
        eff_pred = nn.forward( t )
        eff_target = test_data[i,4]
        eff_clus_target = test_data[i,5]
        eff_clus_pred = 1
        print(f"Predicted DEA score: {eff_pred[0,0]}\nTarget DEA score: {eff_target}\nCluster bound: {eff_clus_pred}\nTarget Cluster: {eff_clus_target}")
    print("--------------------------------------------------------------------------------------------------------------------------------")



def main():
    #buildMixedData()
    #buildDataForAnn()
    nn = runANN(split=0.8,seed=None,num_runs=1,batch_size=1,target_param=False,epochs=200)
    #runANN(split=0.8,seed=None,num_runs=10,batch_size=16)
    #runANN(split=0.8,seed=None,num_runs=10,batch_size=32)
    
    #runANN(split=0.8,seed=None,num_runs=10,batch_size=1,eff_center=eff_mixed_center_name_4,data=mixed_transform_4)
    #runANN(split=0.8,seed=None,num_runs=10,batch_size=16,eff_center=eff_mixed_center_name_4,data=mixed_transform_4)
    #runANN(split=0.8,seed=None,num_runs=10,batch_size=32,eff_center=eff_mixed_center_name_4,data=mixed_transform_4)
    testBestNN(nn,ran_num_sample = 50)
    
    #analyseClusterDistribution()

    #test()
    



if __name__ == "__main__":
    main()