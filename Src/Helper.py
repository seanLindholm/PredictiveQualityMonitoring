import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as isoF
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
from sklearn import preprocessing



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
approved_file_name = data_path + "\\Approved_w_glu.csv"
failed_file_name = data_path + "\\Failed_w_glu.csv"
approved_file_Transform_name = data_path + "\\Approved_w_glu_Transform.csv"
failed_file_Transform_name = data_path + "\\Failed_w_glu_Transform.csv"


# pca_2 = PCA(n_components = 2)
# pca_2.fit(standardized(f_output))
# print("------------------------------------------")
# print("Standardized - PCA: ",end='');print_array(pca_2.explained_variance_ratio_)
# f_pca2_std = pca_2.fit_transform(standardized(f_output))
# pca_2.fit(normalized(f_output))
# print("------------------------------------------")
# print("Normalized - PCA: ",end='');print_array(pca_2.explained_variance_ratio_)
# print("------------------------------------------")
# f_pca2_norm = pca_2.fit_transform(normalized(f_output))


class Data_handler:
    def __init__(self,file_path_csv):
        """
            file_path: a direct or indirect path the to file with the whole name, including
                       the .csv extension\n
            self.dt: The datatabel to be manipulated with\n
            self.colMap: A mapping between column names and the indcies. Can be seen by printing the class
                         or by setting help=True in the function removeColumns\n
        """
        self.__path = file_path_csv
        dt_t = pd.read_csv(file_path_csv,sep=r'\s*,\s*',engine='python')
        self.__normalized = False
        self.colMap = np.array([c[1] for c in enumerate(dt_t.columns)])
        self.dt = dt_t.to_numpy()
    
    def str_support(self):
        pass

    def setPath(self,name_path):
        self.__path = name_path

    def __str__(self):
        return f"Datatabel has {self.dt.shape[0]} samples with {self.dt.shape[1]} columns\nThe column names that can be used as keys for the column mapping are as follows:\n{ self.colMap }"

    def __repr__(self):
        return f"Data analysis from path {self.__path}"

    def removeColumns(self,col_name_list,h=False):
        """
            Used to alter the datatabel by removing unwanted 
            columns. Use the option h=True, if you want to see what
            the column names of the datatable is. Setting help till true
            will not alter the datatabel.
        """
        if h:
            print("Available column name in the datatable:\n")
            for name in self.colMap:
                print(f"{name}")
        else:
            indexis = [np.where(self.colMap == name) for name in col_name_list]
            self.colMap = np.delete(self.colMap,indexis,0)
            self.dt = np.delete(self.dt,indexis,1)
    
    
    def normalized(self):
        """
            Normalizes the datatable in the range (0-1) 
            making use of the min value and peek-to-peek range (max-min)
            and the minimum
        """
        self.org_dt_min =self.dt.min(0)
        self.org_dt_ptp = self.dt.ptp(0)
        self.dt = (self.dt - self.org_dt_min) / self.org_dt_ptp
        self.__normalized = True
    
    def un_normalized(self):
        """
            This function restore the the normalized data tabel into
            the un-normalized version.
        """
        if self.__normalized:
            self.dt = self.dt*self.org_dt_ptp+self.org_dt_min
            self.__normalized = False
        
    def splitData(self,split_indx):
        """
            This is used to split input from output. Only use this function after you have done 
            all the data manipulation you want.\n
            self.dt_in: Ranged from [0:split_indx[\n
            self.dt_out:Ranged from [split_indx:end]\n
        """
        self.dt_in = self.dt[:,:split_indx]
        self.dt_out = self.dt[:,split_indx:]

    # ------ Function for data manupulation ------- #
    
    def standardized(self):
        scale = preprocessing.StandardScaler().fit(self.dt)
        return scale.fit_transform(self.dt)


    def outlierRemoval(self,contamination=0.1):
        """ 
            This outlier detector makes use of the Isolated tree method, from
            the sklearn library.
        """
        iso = isoF(contamination=contamination)
        yhat = iso.fit_predict(self.dt)
        mask = yhat != -1
        self.dt = self.dt[mask,:]
        return 

    def kMean(self,dt,num_clus,seed=0):
        """ 
         This function utilises sklearn's implementation
         of the kMeans algortihmn
        """
        return km(n_clusters=num_clus, random_state=seed).fit(dt)
    
    def lowestComponancePCA(self,min_comp,explain_var,output_only = True):
        """
            This function builds finds the lowest number of Principal Components
            that is equal to or above the specified explain_variance (range 0-1)
        """
        if (output_only):
            if self.dt_out == []: 
                print("You haven't split the data in input-output yet!")
                return 
            else:
                for i in range(min_comp,self.dt_out.shape[1]+1):
                    pca = PCA(n_components=i)
                    pca.fit(self.dt_out)
                    if(sum(pca.explained_variance_ratio_) >= explain_var):
                        return pca
                return pca
        else:
            for i in range(min_comp,self.dt.shape[1]+1):
                pca = PCA(n_components=i)
                pca.fit(self.dt)
                if(sum(pca.explained_variance_ratio_) >= explain_var):
                    return pca
            return pca





















# TODO:
"""
  *Transform data to the off-set of the bounds (absolut) (the center of the bounds) - assumption
     -Techinal question is below the same as above the bound (Question for company)
         -In case not the same, we can use weighting 
  *All distance says how bad -> lable the samples (range of badness)
  *Drop parameters without bounds
  *Perfomance evaluation (DEA - Data envelopment analysis) know sample is measured with x param each has value, the higher the worse

  *Distance -> performance index for sample. DEA gives unique number for each sample. From x -> 1 score
  *pyDEA - > library: http://people.brunel.ac.uk/~mastjjb/jeb/or/dea.html
  *Be able to show the result is robust to the method, by showing simluar results from different methods
  *Understading of the underlying method - exam. 


"""






