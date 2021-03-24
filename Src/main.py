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
# These four paths are created when buildDataForann has run
eff_mixed_center_name_5 = data_path + "\\dea_eff_centroid5_transformed.csv"
eff_mixed_center_name_4 = data_path + "\\dea_eff_centroid4_transformed.csv"
mixed_transform_5 = data_path + "\\mixed_with_clusters5_transformed.csv"
mixed_transform_4 = data_path + "\\mixed_with_clusters4_transformed.csv"

# These two paths are created when buildDataForann has run
mixed_5 = data_path + "\\mixed_with_clusters5.csv"
mixed_4 = data_path + "\\mixed_with_clusters4.csv"
cluster_center_5 = data_path + "\\cluster_centroid5.csv"
cluster_center_4 = data_path + "\\cluster_centroid4.csv"

# https://stackoverflow.com/questions/4493554/neural-network-always-produces-same-similar-outputs-for-any-input

def buildMixedData(approved_data = approved_file_Transform_name, failed_data = failed_file_Transform_name,transformed=True):

    # Build mixed class from approved and failed data
    dh_approved = Data_handler(approved_data)
    dh_failed = Data_handler(failed_data)
    dh_approved.dt[:,3] = 1
    dh_failed.dt[:,3] = 0

    if not transformed:
        dh_approved.removeColumns(['Tid i min Glu'])
        dh_approved.deleteSavedData()
        dh_failed.removeColumns(['Tid i min Glu'])
        dh_failed.deleteSavedData()
    #Frist remove all outliers
    dh_approved.outlierRemoval(contamination=0.01)
    dh_failed.outlierRemoval(contamination=0.01)
    #Now make sure that there is equal amount of approved/failed arrays
    Ran_app_sel_trans = np.random.permutation(np.arange(dh_approved.dt.shape[0]))[:dh_failed.dt.shape[0]]
    dh_approved.dt = dh_approved.dt[Ran_app_sel_trans,:]

    #Construct the mixed clas from the two data_handlers
    if transformed:
        dh_mixed = Data_handler.from2DH(dh_approved,dh_failed)
        dh_mixed.saveToCsv('Mixed_transform_withClass_unNorm')
        dh_mixed.removeColumns(['Class'])
        dh_mixed.saveToCsv('Mixed_transform_noClass_unNorm')

        dh_mixed.normalized()
        dh_mixed.saveToCsv('Mixed_transform_noClass_normalized')
        dh_mixed.restoreSavedData()
        dh_mixed.saveToCsv('Mixed_transform_WithClass_normalized')
    else:
        dh_mixed = Data_handler.from2DH(dh_approved,dh_failed)
        dh_mixed.saveToCsv('Mixed_withClass_unNorm')
        dh_mixed.removeColumns(['Class'])
        dh_mixed.saveToCsv('Mixed_noClass_unNorm')

        dh_mixed.normalized()
        dh_mixed.saveToCsv('Mixed_noClass_normalized')
        dh_mixed.restoreSavedData()
        dh_mixed.saveToCsv('Mixed_WithClass_normalized')



    
def buildDataForAnn(mixed_data_withClass_norm ="..\\Data\\Mixed_transform_withClass_normalized.csv",mixed_data_noClass_norm = "..\\Data\\Mixed_transform_noClass_normalized.csv",mixed_data_withClass = "..\\Data\\Mixed_transform_withClass_unNorm.csv",transformed=True):
    
    #Always build ANN_data_5cluster.csv like this:

    
    dh_mixed = Data_handler(file_path_csv=mixed_data_withClass_norm)
    dh_mixed.removeColumns(['Class'])

    dh_mixed.splitData(3)
    dh_mixed_eff = Data_handler(file_path_csv=eff_mixed_name)
    labels = printOutput([dh_mixed,dh_mixed_eff],start_cluster=4)
    class_eff5 = Data_handler.dataAndHeader(labels[1][2],["cluster efficency - 5 clusters"])
    class_norm5 = Data_handler.dataAndHeader(labels[0][2],["cluster normalized - 5 clusters"])

    dh_mixed.restoreSavedData()
   
    dh_mixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','Sensitivity [pA/µM]','t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2'])


    dh_normMixed = Data_handler(file_path_csv=mixed_data_noClass_norm)
    dh_unnormMixed = Data_handler(file_path_csv=mixed_data_withClass)
    if transformed:
        dh_paramTransform_norm = Data_handler.dataAndHeader(dh_normMixed.dt[:,3:],['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
        dh_unnormMixed.append(dh_mixed_eff,axis=1)
        dh_unnormMixed.append(class_eff5,axis=1)
        dh_unnormMixed.append(class_norm5,axis=1)
        dh_mixed.append(dh_mixed_eff,axis=1)
        dh_mixed.append(class_eff5,axis=1)
        dh_mixed.append(class_norm5,axis=1)
        dh_mixed.restoreSavedData()
        dh_mixed.saveToCsv("ANN_data_5cluster")


    else:
        dh_paramTransform_norm = Data_handler.dataAndHeader(dh_normMixed.dt[:,3:],['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','40/25 mM glu/lac lav O2','Sensitivity [pA/µM],t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2,50 mM Mannose','35 mM Glycolsyre,2 mM PAM'])
        dh_unnormMixed.append(class_norm5,axis=1)

  
    if transformed:
        dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    else:
        dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','40/25 mM glu/lac lav O2','Sensitivity [pA/µM]','t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2','50 mM Mannose','35 mM Glycolsyre','2 mM PAM'])
    dh_unnormMixed.deleteSavedData()
    dh_unnormMixed.append(dh_paramTransform_norm,axis=1)


    if not transformed:
        dh_unnormMixed.saveToCsv('mixed_with_clusters5')
        centroid = Data_handler.dataAndHeader(labels[0][3],['2/1 mM Glu/Lac [mM] - center(norm)','1 mM H2O2 [mM] - center(norm)','40/25 mM glu/lac høj O2 - center(norm)','40/25 mM glu/lac lav O2 - center(norm)','Sensitivity [pA/µM] - center(norm)','t on 10/5 mM glu/lac [s] - center(norm)','Lav O2 - Høj O2 - center(norm)','50 mM Mannose - center(norm)','35 mM Glycolsyre - center(norm)','2 mM PAM - center(norm)'])
        centroid.saveToCsv('cluster_centroid5')
    else:
        dh_unnormMixed.saveToCsv('mixed_with_clusters5_transformed')
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
    
        dea_eff_centroid.saveToCsv('dea_eff_centroid5_transformed')



    class_eff4 = Data_handler.dataAndHeader(labels[1][0],["cluster efficency - 4 clusters"])
    class_norm4 = Data_handler.dataAndHeader(labels[0][0],["cluster normalized - 4 clusters"])

    dh_normMixed = Data_handler(file_path_csv=mixed_data_noClass_norm)
    dh_unnormMixed = Data_handler(file_path_csv=mixed_data_withClass)
    if transformed:
        dh_paramTransform_norm = Data_handler.dataAndHeader(dh_normMixed.dt[:,3:],['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
        dh_unnormMixed.append(dh_mixed_eff,axis=1)
        dh_unnormMixed.append(class_eff4,axis=1)
        dh_unnormMixed.append(class_norm4,axis=1)
    else:
        dh_paramTransform_norm = Data_handler.dataAndHeader(dh_normMixed.dt[:,3:],['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','40/25 mM glu/lac lav O2','Sensitivity [pA/µM],t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2,50 mM Mannose','35 mM Glycolsyre,2 mM PAM'])
        dh_unnormMixed.append(class_norm4,axis=1)

  
    if transformed:
        dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    else:
        dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','40/25 mM glu/lac lav O2','Sensitivity [pA/µM]','t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2','50 mM Mannose','35 mM Glycolsyre','2 mM PAM'])

    dh_unnormMixed.deleteSavedData()
    dh_unnormMixed.append(dh_paramTransform_norm,axis=1)
   
    if not transformed:
        dh_unnormMixed.saveToCsv('mixed_with_clusters4')
        centroid = Data_handler.dataAndHeader(labels[0][1],['2/1 mM Glu/Lac [mM] - center(norm)','1 mM H2O2 [mM] - center(norm)','40/25 mM glu/lac høj O2 - center(norm)','40/25 mM glu/lac lav O2 - center(norm)','Sensitivity [pA/µM] - center(norm)','t on 10/5 mM glu/lac [s] - center(norm)','Lav O2 - Høj O2 - center(norm)','50 mM Mannose - center(norm)','35 mM Glycolsyre - center(norm)','2 mM PAM - center(norm)'])
        centroid.saveToCsv('cluster_centroid4')

    else:
        dh_unnormMixed.saveToCsv('mixed_with_clusters4_transformed')


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
        dea_eff_centroid.saveToCsv('dea_eff_centroid4_transformed')
      




def test():
    dh = Data_handler("..\\Data\\dummy.csv")
    dh2 = Data_handler(mixed_transform_5)
    
    print(np.mean(dh2.dt[:,4]))
    print()
    dh2.normalized()
    dh2.splitData(3)
    dh.splitData(3)
    print(np.std(dh.dt_in))
    print(np.var(dh.dt_in))
    print()
    print(np.std(dh2.dt_in))
    print(np.var(dh2.dt_in))
    dh_2 = Data_handler("..\\Data\\Mixed_transform_withClass_normalized.csv")
    dh_2.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]','40/25 mM glu/lac høj O2','Sensitivity [pA/µM]','t on 10/5 mM glu/lac [s]','Lav O2 - Høj O2','Class'])
    dh_2.deleteSavedData()
    dh_3 = Data_handler("..\\Data\\mixed_with_clusters5_transformed.csv")
    dh_3.removeColumns(['time_betw_scan_hours','CA','YM'])
    dh_2.append(dh_3,axis=1)
    dh_2.saveToCsv('test_data')
    


    




def runANN(split,seed,num_runs=1,batch_size = 8,epochs = 200,eff_center=eff_mixed_center_name_5,data="..\\Data\\ANN_data_5cluster.csv",class_prediction=False,early_stopping=True,in_features=3,split_indx = 3):
    #This will be used as part of the cost-function
    dh_eff_cent = Data_handler(file_path_csv=eff_center)
    
    #This is the data that needs to be trained on
    dh_data = Data_handler(file_path_csv=data)
    
    #Split the data in in_ out_ and select the dea_eff score as label
    dh_data.splitData(split_indx)
    X,y = torch.tensor(dh_data.dt_in),torch.tensor(dh_data.dt_out)
    acc = 0
    acc_best = -1
    best_nn = None
    for i in range(num_runs):
        progressBar(i,num_runs)

        #Split data in train and test with 80 % train and 20 % test
        X_train,y_train,X_test,y_test = splitData(X,y,proc_train=split,seed=seed)    
        X_train,y_train,X_test,y_test = createBatch(X_train,y_train,X_test, y_test,batch_size=batch_size)
        nn = Net(in_features = in_features,dh_classTarget=dh_eff_cent,class_prediction=class_prediction,early_stopping=early_stopping).to(device)


        #Construct the network with the appropiate number of input data for each sample
        acc_tmp = nn.train(X_train,y_train,X_test,y_test,epochs=epochs)
        if (acc_tmp > acc_best):
            acc_best = acc_tmp
            best_nn = nn 
        acc += acc_tmp
        progressBar(i+1,num_runs)
    
    print(f"Done traning with {num_runs} runs. The average accuracy is scored as {acc/num_runs}.")
    return best_nn

def analyseClusterDistribution(Mixed5 = mixed_transform_5, Mixed4 = mixed_transform_4, eff5 = eff_mixed_center_name_5, eff4 = eff_mixed_center_name_4, transformed=True):
    dh_unnormMixed = Data_handler(file_path_csv=Mixed5)
    dh_unnormMixed_4 = Data_handler(file_path_csv=Mixed4)

    dh_classTarget = Data_handler(file_path_csv=eff5)
    dh_classTarget_4 = Data_handler(file_path_csv=eff4)
    if transformed:
       

        

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
    else:
      

        #Now find std for the different clusters of eff
        mask_0 = np.where(dh_unnormMixed.dt[:,4] == 0)
        mask_1 = np.where(dh_unnormMixed.dt[:,4] == 1)
        mask_2 = np.where(dh_unnormMixed.dt[:,4] == 2)
        mask_3 = np.where(dh_unnormMixed.dt[:,4] == 3)
        mask_4 = np.where(dh_unnormMixed.dt[:,4] == 4)

     
        
        print("Class distribtuion with 5 clusters on normalized distance parameters - None transformed\n")
        print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed.dt[mask_0,3].squeeze())}, centroid: {dh_classTarget.dt[0,:]}")
        print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed.dt[mask_1,3].squeeze())}, centroid: {dh_classTarget.dt[1,:]}")
        print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed.dt[mask_2,3].squeeze())}, centroid: {dh_classTarget.dt[2,:]}")
        print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed.dt[mask_3,3].squeeze())}, centroid: {dh_classTarget.dt[3,:]}")
        print(f"Class distribution in cluster 4: {Counter(dh_unnormMixed.dt[mask_4,3].squeeze())}, centroid: {dh_classTarget.dt[4,:]}\n")

         #Now find std for the different clusters of eff
        mask_0 = np.where(dh_unnormMixed_4.dt[:,4] == 0)
        mask_1 = np.where(dh_unnormMixed_4.dt[:,4] == 1)
        mask_2 = np.where(dh_unnormMixed_4.dt[:,4] == 2)
        mask_3 = np.where(dh_unnormMixed_4.dt[:,4] == 3)
    

        
        print("Class distribtuion with 4 clusters on normalized distance parameters - None transformed\n")
        print(f"Class distribution in cluster 0: {Counter(dh_unnormMixed_4.dt[mask_0,3].squeeze())}, centroid: {dh_classTarget_4.dt[0,:]}")
        print(f"Class distribution in cluster 1: {Counter(dh_unnormMixed_4.dt[mask_1,3].squeeze())}, centroid: {dh_classTarget_4.dt[1,:]}")
        print(f"Class distribution in cluster 2: {Counter(dh_unnormMixed_4.dt[mask_2,3].squeeze())}, centroid: {dh_classTarget_4.dt[2,:]}")
        print(f"Class distribution in cluster 3: {Counter(dh_unnormMixed_4.dt[mask_3,3].squeeze())}, centroid: {dh_classTarget_4.dt[3,:]}\n")

        



def testBestNN(nn,data="..\\Data\\ANN_data_5cluster.csv",target=eff_mixed_center_name_5,target_param=False,ran_num_sample=10,split_indx = 3,eff_target_indx = 4,eff_clus_target_indx = 5):
    dh = Data_handler(file_path_csv=data)
    dh_t = Data_handler(file_path_csv=target)

    Ran_indecies = np.random.permutation(np.arange(dh.dt.shape[0]))
    test_data = dh.dt[Ran_indecies[:ran_num_sample]]

    for i in range (ran_num_sample):
        t = torch.FloatTensor([np.array(test_data[i,:split_indx])]).cuda()
        print(t)
        eff_pred = nn.forward( t )
        eff_target = test_data[i,eff_target_indx]
        eff_clus_target = test_data[i,eff_clus_target_indx]
        eff_clus_pred = 1
        print(f"Predicted DEA score: {eff_pred[0,0]}\nTarget DEA score: {eff_target}\nCluster bound: {eff_clus_pred}\nTarget Cluster: {eff_clus_target}")
    nn.plot()
    print("-----------------------------------------------------------------------------------------------------------------")



def main():
    #buildMixedData(approved_data = approved_file_Transform_name, failed_data = failed_file_Transform_name,transformed=True)
    #buildMixedData(approved_data = approved_file_name, failed_data = failed_file_name,transformed=False)

    #buildDataForAnn(mixed_data_withClass_norm ="..\\Data\\Mixed_transform_withClass_normalized.csv",mixed_data_noClass_norm = "..\\Data\\Mixed_transform_noClass_normalized.csv",mixed_data_withClass = "..\\Data\\Mixed_transform_withClass_unNorm.csv",transformed=True)
    #buildDataForAnn(mixed_data_withClass_norm ="..\\Data\\Mixed_withClass_normalized.csv",mixed_data_noClass_norm = "..\\Data\\Mixed_noClass_normalized.csv",mixed_data_withClass = "..\\Data\\Mixed_withClass_unNorm.csv",transformed=False)

    # print("-------------------- DUMMY TEST ------------------------------")
    # # Need to change Ann.py to run this. In training set y[:,1] to y, and outcomment accuracy calculation
    # nn = runANN(split=1,seed=None,num_runs=5,batch_size=2,class_prediction=False,epochs=500,early_stopping=False,eff_center=eff_mixed_center_name_4,data="..\\Data\\dummy.csv")
    # testBestNN(nn,ran_num_sample = 6,data="..\\Data\\dummy.csv",eff_target_indx=3,eff_clus_target_indx=1)
    # print("-------------------DUMMY TEST END-----------------------------")

    
    #nn = runANN(split=0.8,seed=None,num_runs=5,batch_size=32,class_prediction=False,epochs=500,early_stopping=False)
    #testBestNN(nn,ran_num_sample = 15)

    #nn = runANN(split=0.8,seed=None,num_runs=5,batch_size=64,class_prediction=False,epochs=500,early_stopping=False)
    #testBestNN(nn,ran_num_sample = 15)
    
    #nn = runANN(split=0.8,seed=None,num_runs=5,batch_size=128,class_prediction=False,epochs=500,early_stopping=False)
    #testBestNN(nn,ran_num_sample = 15)
    
    
    analyseClusterDistribution(Mixed5 = mixed_transform_5, Mixed4 = mixed_transform_4, eff5 = eff_mixed_center_name_5, eff4 = eff_mixed_center_name_4, transformed=True)
    #analyseClusterDistribution(Mixed5 = mixed_5, Mixed4 = mixed_4, eff5=cluster_center_5,eff4=cluster_center_4, transformed=False)

    # test()
    



if __name__ == "__main__":
    main()