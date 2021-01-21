
from Helper import *
from collections import Counter
import matplotlib.pyplot as plt

def FindMaxEuclidDistFromSet(L_dt,center_matrix):
    ret = {}
    center_indx = 0
    for dt in L_dt:
        max_dist = 0
        for row in dt:
            dist = np.linalg.norm(row-center_matrix[center_indx,:])
            if (max_dist < dist):
                max_dist = dist
        ret[center_indx] = max_dist
        center_indx+=1
    print("\nThe maximum euclidian distance of an element to the closets centroid")
    print(ret)
    return ret


def kMeansWithLabels(dt,n_clusters,seed):
    L_dt = []
    KMs = Data_handler.kMean(Data_handler,dt,n_clusters,seed)
    cluster_label = np.transpose(np.array([KMs.labels_]))
    f_out = np.append(dt,cluster_label,axis=1)
    # Print the number of points in the centers along the centers themself
    print(f"The cluster centroids for {n_clusters} clusters:")
    print_matrix(KMs.cluster_centers_)
    print("\nNumber of elements in each cluster:")
    print(Counter(KMs.labels_))

    # devide the output into the labels using masks
    for i in range(n_clusters):
        mask = f_out[:,-1] == i
        L_dt.append(f_out[mask,:-1])

    # Return the matrix and the Matrix of matrixes where it has been divided 
    return f_out,L_dt,KMs.cluster_centers_

def testAndPrintUpToNClusters(dt,n_clusters,seed):
    if n_clusters < 2: n_clusters = 2; 
    for i in range(2,n_clusters+1):
        _,L_dt,center_matrix = kMeansWithLabels(dt,i,seed)
        _ = FindMaxEuclidDistFromSet(L_dt,center_matrix)
        print()

def pcaClosterPlot(dt,clusters):
    if clusters > 0:
        f_o,_,center_matrix = kMeansWithLabels(dt,clusters,1337)
        c_mask = np.zeros(f_o.shape[0])
        for i in range(clusters):
            mask = f_o[:,-1] == i
            c_mask += mask*(i+1)

        plt.scatter(f_o[:,0],f_o[:,1],c=c_mask,alpha=0.4,marker='.')
        plt.scatter(center_matrix[:,0],center_matrix[:,1],c='black',marker='+',alpha=1,s=100)
        plt.title(f"marked clusters with {clusters} number of clusters.")
    else:
       
        plt.scatter(dt[:,0],dt[:,1],marker='.')
        plt.title("data with 2 principal components")


def printOutput(dh_list):
    for dh in dh_list:
        print()
        print(dh.__repr__())
        print()
        pca_red = dh.lowestComponancePCA(min_comp=2,explain_var = 0.95,output_only = True).fit_transform(dh.dt_out)
        print("Use of Kmeans clustering after removal of outliers with NO reduction of dimention - Normalized:\n")
        testAndPrintUpToNClusters(dh.dt_out,3,0)#f_output.shape[1],1337)
        #print("Use of Kmeans clusterng after removal of outliers with NO reduction of dimention - Standardized:\n")
        #testAndPrintUpToNClusters(standardized(f_output),4,1337)#f_output.shape[1],1337)
        print("\n\n\nUse of Kmeans clustering after removal of outliers with PCA reduction:\n")
        testAndPrintUpToNClusters(pca_red,3,0)
    

    # print ("Use of kMeans clustering on Transformed dataset - normalized")
    
    # # Show plot of a 2D - pca clustering
    # plt.figure('No clusters - norm')
    # pcaClosterPlot(f_pca2_norm,0)
    # plt.figure('Two clusters - norm')
    # pcaClosterPlot(f_pca2_norm,2)
    # plt.figure('Three clusters - norm')
    # pcaClosterPlot(f_pca2_norm,3)


    # # Show plot of a 2D - pca clustering
    # plt.figure('No clusters - std')
    # pcaClosterPlot(f_pca2_std,0)
    # plt.figure('Two clusters - std')
    # pcaClosterPlot(f_pca2_std,2)
    # plt.figure('Three clusters - std')
    # pcaClosterPlot(f_pca2_std,3)

    plt.show(block=False)
    input("Press Enter to close the figures...")
    plt.close('all')

    


def main():
    dh_l = [Data_handler(approved_file_name),Data_handler(failed_file_name),Data_handler(failed_file_name)]
    dh_l_transform = [Data_handler(approved_file_Transform_name),Data_handler(failed_file_Transform_name),Data_handler(failed_file_Transform_name)]
    
    for dh in dh_l:
        dh.removeColumns(['Class','Tid i min Glu'])
        
    for dh in dh_l_transform:
        dh.removeColumns(['Class'])

    for elm in dh_l_transform:
        dh_l.append(elm)
    
    Ran_app_sel = np.random.permutation(np.arange(dh_l[0].dt.shape[0]))[:dh_l[1].dt.shape[0]]
    Ran_app_sel_trans = np.random.permutation(np.arange(dh_l[3].dt.shape[0]))[:dh_l[4].dt.shape[0]]
    dt_mixed = np.append(dh_l[1].dt,dh_l[0].dt[Ran_app_sel,:],axis=0)
    dt_mixed_trans = np.append(dh_l[4].dt,dh_l[3].dt[Ran_app_sel_trans,:],axis=0)

    dh_l[2].dt = dt_mixed
    dh_l[2].setPath("mixed tabel with approved and failed")
    dh_l[5].dt = dt_mixed_trans
    dh_l[5].setPath("mixed tabel with approved and failed - Transformed")


    for dh in dh_l:
        dh.outlierRemoval()
        df = dh.dt
        dh.normalized()
        dh.splitData(3)

    printOutput(dh_l)


    
if __name__ == "__main__":
    main()
   