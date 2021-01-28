
from Helper import *
from collections import Counter

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
    prettyprint.print_matrix(KMs.cluster_centers_)
    print("\nNumber of elements in each cluster:")
    print(Counter(KMs.labels_))
    # devide the output into the labels using masks
    for i in range(n_clusters):
        mask = f_out[:,-1] == i
        L_dt.append(f_out[mask,:-1])

    # Return the matrix and the Matrix of matrixes where it has been divided 
    return f_out,L_dt,KMs.cluster_centers_,cluster_label

def testAndPrintUpToNClusters(dt,n_clusters,seed,start_cluster=2):
    ret = []
    if n_clusters < 2: n_clusters = 2
    if start_cluster > n_clusters: n_clusters = start_cluster+1
    for i in range(start_cluster,n_clusters+1):
        _,L_dt,center_matrix,cl_label = kMeansWithLabels(dt,i,seed)
        _ = FindMaxEuclidDistFromSet(L_dt,center_matrix)
        ret.append(cl_label)
        ret.append(center_matrix)
        print()
    return ret

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


def printOutput(dh_list,start_cluster=2,include_pca=False):
    ret = []
    for dh in dh_list:
        print()
        print(dh.__repr__())
        print()
        pca_red = dh.lowestComponancePCA(min_comp=2,explain_var = 0.95,output_only = True).fit_transform(dh.dt_out)
        print("Use of Kmeans clustering after removal of outliers with NO reduction of dimention - Normalized:\n")
        ret_test = testAndPrintUpToNClusters(dh.dt_out,6,0,start_cluster)#f_output.shape[1],1337)
        #print("Use of Kmeans clusterng after removal of outliers with NO reduction of dimention - Standardized:\n")
        #testAndPrintUpToNClusters(standardized(f_output),4,1337)#f_output.shape[1],1337)
        print("\n\n\nUse of Kmeans clustering after removal of outliers with PCA reduction:\n")
        ret_pca = testAndPrintUpToNClusters(pca_red,6,0,start_cluster)
        ret.append(ret_test)
        if(include_pca):
            ret.append(ret_pca)
    

    return ret

    



   