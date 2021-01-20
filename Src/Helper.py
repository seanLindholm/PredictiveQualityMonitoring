import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as isoF
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA


# ------ Function for data manupulation ------- #

def normalized(dt):
    return (dt - dt.min(0)) / dt.ptp(0)
    

def standardize_data(dt):
    pass

def outlierRemoval_mask(dt):
    # This outlier detector makes use of the Isolated tree, from
    # sklearn library
    iso = isoF(contamination=0.1)
    yhat = iso.fit_predict(dt)
    return yhat != -1

def kMean(dt,num_clus,seed=0):
    # This function utilises sklearn's implementation
    # of the kMeans algortihmn
    return km(n_clusters=num_clus, random_state=seed).fit(dt)

def lowestComponancePCA(dt,min_comp,explain_var):
    for i in range(min_comp,dt.shape[1]+1):
        pca = PCA(n_components=i)
        pca.fit(dt)
        if(sum(pca.explained_variance_ratio_) >= explain_var):
            print(sum(pca.explained_variance_ratio_))
            return pca
    return pca
# -------------------------------------------- #


# ----- Function for numpy pretty print ------ #
def print_matrix(M,header=[],float_precision=3):
    if (header != []):
        pass                                                                                                                                                                                                                                            
    for row in M:
        print("|",end='')
        for v in row:
            print("{0:>7.{1}f} ".format(v,float_precision),end='')
        print("|") 
  

def print_array(A,float_precision=3):
    print("|",end='')
    for v in A:
        print("{0:>7.{1}f} ".format(v,float_precision),end='')
    print("|")   
 

# -------------------------------------------- #


#The path to the data used
data_path = "..\\Data"
approved_file_name = data_path + "\\Approved_w_glu_3000rows_15_1_2021.csv"
failed_file_name = data_path + "\\Failed_w_glu_15_1_2021.csv"


#Reading in data and removing the onwanted columns
approved_data = pd.read_csv(approved_file_name,sep=r'\s*,\s*',engine='python')
col_mapping_dict = {c[1]:c[0] for c in enumerate(approved_data.columns)}
approved_data = approved_data.to_numpy()
failed_data = pd.read_csv(failed_file_name,sep=r'\s*,\s*',engine='python').to_numpy()

#For now we remove the 'Tid efter start [timer]' and the 'Tid in GLU' since these values are due to human handling 
#and not a measure of the product
approved_data = np.delete(approved_data,[col_mapping_dict['Tid efter start [timer]'],col_mapping_dict['Tid i min Glu']],1)
failed_data = np.delete(failed_data,[col_mapping_dict['Tid efter start [timer]'],col_mapping_dict['Tid i min Glu']],1)

# We here also avoid the class column since this is just binary good/bad 
f_input,f_output = failed_data[:,:3],failed_data[:,4:]


# We now remove outliers in the output - the reason that we do it in the output is that the outlier is not 
# Tied to the input but rather the the machine worked probaly when measuring. This can also be tied to 
# the resolution pack not working proberly, or being to old.
mask = outlierRemoval_mask(f_output)
f_input,f_output = f_input[mask,:], f_output[mask,:]


# Same for the approved data, alongside a reduction such that we got half and half 
a_input,a_output = approved_data[:,:3],approved_data[:,4:]
mask = outlierRemoval_mask(a_output)
a_input,a_output = a_input[mask,:], a_output[mask,:]


# Now build the f_output from a reduced set (PCA) after outlier removal
# Find the lowest number of components that still explain 95% variance
pca = lowestComponancePCA(dt = normalized(f_output),min_comp=2,explain_var = 0.95)
f_output_pca = pca.fit_transform(normalized(f_output))


pca_2 = PCA(n_components = 2)
pca_2.fit(normalized(f_output))
f_pca2 = pca_2.fit_transform(normalized(f_output))


