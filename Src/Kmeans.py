
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
    KMs = kMean(dt,n_clusters,seed)
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
    for i in range(1,n_clusters+1):
        _,L_dt,center_matrix = kMeansWithLabels(dt,i,seed)
        _ = FindMaxEuclidDistFromSet(L_dt,center_matrix)
        print()

def pcaClosterPlot(clusters):
    if clusters > 0:
        f_o,_,center_matrix = kMeansWithLabels(f_pca2,clusters,1337)
        c_mask = np.zeros(f_o.shape[0])
        for i in range(clusters):
            mask = f_o[:,-1] == i
            c_mask += mask*(i+1)

        plt.scatter(f_o[:,0],f_o[:,1],c=c_mask,alpha=0.4,marker='.')
        plt.scatter(center_matrix[:,0],center_matrix[:,1],c='black',marker='+',alpha=1,s=100)
        plt.title(f"marked clusters with {clusters} number of clusters.")
    else:
       
        plt.scatter(f_pca2[:,0],f_pca2[:,1],marker='.')
        plt.title("data with 2 principal components")


def main():
    print("Use of Kmeans clustering after removal of outliers with NO reduction of dimention:\n")
    testAndPrintUpToNClusters(normalized(f_output),f_output.shape[1],1337)
    print("\n\n\nUse of Kmeans clustering after removal of outliers with PCA reduction:\n")
    testAndPrintUpToNClusters(f_output_pca,f_output_pca.shape[1],1337)
  
    # Show plot of a 2D - pca clustering
    plt.figure('No clusters')
    pcaClosterPlot(0)
    plt.figure('Two clusters')
    pcaClosterPlot(2)
    plt.figure('Three clusters')
    pcaClosterPlot(3)
    plt.figure('Four clusters')
    pcaClosterPlot(4)

    plt.show(block=False)
    input("Press Enter to close the figures...")
    plt.close('all')

    # Maybe use the DBSCAN to find density clusters
if __name__ == "__main__":
    main()
   