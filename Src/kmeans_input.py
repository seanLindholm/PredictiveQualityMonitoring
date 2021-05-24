from constants_ import *
from sklearn.cluster import KMeans as km
from Helper import prettyprint

def clusterInformation(label,class_,max_f,max_a,clusters):
    class_[class_ == 'Approved'] = 1
    class_[class_ == 'Failed'] = 0
    dict_ = {}
    for i in range(label.shape[0]):
        if not label[i] in dict_:
            dict_[label[i]] = [0,0]
        dict_[label[i]][class_[i]] += 1

    print(dict_)
    for label in range(clusters):
        print(f"Cluster {label} has {(dict_[label][0]/max_f)*100:.2f}% of total number of failed and {(dict_[label][1]/max_a)*100:.2f}% of total number of approved arrays")
    print()

def main(num_clusters=2):
    print("-----------------------------------------------------------------------------")
    print(f"{num_clusters} number of clusters without picture slip, and with slip pictures")
    df = getData(failed_NoNaN)
    max_failed = df.shape[0]
    df_ = getData(approved_NoNaN)
    max_approved = df_.shape[0]

    dt_mixed = normalize(np.append(df[fcnn_data].to_numpy(),df_[fcnn_data].to_numpy(),axis=0))
    dt_mixed_class = np.append(df['Class'].to_numpy().reshape(-1,1),df_['Class'].to_numpy().reshape(-1,1),axis=0)
    clusters = km(n_clusters=num_clusters, random_state=None).fit(dt_mixed)
    cluster_label = np.transpose(np.array([clusters.labels_]))

    clusterInformation(cluster_label.squeeze(),dt_mixed_class.squeeze(),max_failed,max_approved,num_clusters)
    print(f"cluster centers for {num_clusters} clusters")
    prettyprint.print_matrix(clusters.cluster_centers_)
    print()
    pictures_ = loadImageData("numpyData\\img_data_split_YM").reshape(loadImageData("numpyData\\img_data_split_YM").shape[0],-1)
    print("Clustering with the features from the prerimiter of the sensor\n")
    data = normalize(np.append(df.append(df_)[fcnn_data].to_numpy(),pictures_,axis=1))
    clusters = km(n_clusters=num_clusters, random_state=None).fit(data)
    cluster_label = np.transpose(np.array([clusters.labels_]))
    clusterInformation(cluster_label.squeeze(),dt_mixed_class.squeeze(),max_failed,max_approved,num_clusters)
    print()
    print("Clustering the image features by themself\n")
    data = normalize(pictures_)
    clusters = km(n_clusters=num_clusters, random_state=None).fit(data)
    cluster_label = np.transpose(np.array([clusters.labels_]))
    clusterInformation(cluster_label.squeeze(),dt_mixed_class.squeeze(),max_failed,max_approved,num_clusters)


if __name__ == "__main__":
    for clusters in range (2,9):
        main(clusters)
        print("-----------------------------------------------------------------------------")



