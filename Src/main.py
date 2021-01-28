from Helper import Data_handler, prettyprint as pp
from Kmeans import *

#The path to the data used
data_path = "..\\Data"
approved_file_name = data_path + "\\Approved_w_glu.csv"
failed_file_name = data_path + "\\Failed_w_glu.csv"
approved_file_Transform_name = data_path + "\\Approved_w_glu_Transform.csv"
failed_file_Transform_name = data_path + "\\Failed_w_glu_Transform.csv"
eff_failed_name = data_path + "\\eff_failed.csv"
eff_approved_name = data_path + "\\eff_approved.csv"
eff_mixed_name = data_path + "\\eff_mixed.csv"

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
    print(dh_mixed)
    dh_mixed_eff = Data_handler(file_path_csv=eff_mixed_name)
    labels = printOutput([dh_mixed,dh_mixed_eff],5)

    class_norm5 = Data_handler.dataAndHeader(labels[0][0],["cluster normalized - 5 clusters"])
    class_eff5 = Data_handler.dataAndHeader(labels[1][0],["cluster efficency - 5 clusters"])
    dh_mixed.restoreSavedData()


    dh_unnormMixed = Data_handler(file_path_csv="..\\Data\\Mixed_transform_withClass_unNorm.csv")
    dh_unnormMixed.append(dh_mixed_eff,axis=1)
    dh_unnormMixed.append(class_eff5,axis=1)
    dh_unnormMixed.removeColumns(['2/1 mM Glu/Lac [mM]','1 mM H2O2 [mM]', '40/25 mM glu/lac høj O2', 'Sensitivity [pA/µM]', 't on 10/5 mM glu/lac [s]', 'Lav O2 - Høj O2'])
    dh_unnormMixed.saveToCsv('mixed_with_clusters')
    dea_eff_centroid = Data_handler.dataAndHeader(labels[1][1],["Centers - 5 cluster"])
    dea_eff_centroid.saveToCsv('dea_eff_centroid')



def main():
    #buildMixedData()
    buildDataForAnn()
    
    



if __name__ == "__main__":
    main()