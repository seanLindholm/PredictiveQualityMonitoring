import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as isoF
from sklearn.cluster import KMeans as km
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.optimize import fmin_slsqp




class prettyprint:
    # ----- Function for numpy pretty print ------ #
    @staticmethod
    def print_matrix(M,header=[],float_precision=3):
        if (header != []):
            pass                                                                                                                                                                                                                                            
        for row in M:
            print("|",end='')
            for v in row:
                print("{0:>7.{1}f} ".format(v,float_precision),end='')
            print("|") 
    
    @staticmethod
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
eff_failed_name = data_path + "\\eff_failed.csv"
eff_approved_name = data_path + "\\eff_approved.csv"
eff_mixed_name = data_path + "\\eff_mixed.csv"


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
    def __init__(self,file_path_csv = None, colMap = None, dt = None):
        """
            file_path: a direct or indirect path the to file with the whole name, including
                       the .csv extension. When set overrides colMap and dt input\n
            colMap: the Column mapping of the data_tabel\n
            dt: the datatable. If both colMap and dt is set, file_path_csv is ignored\n
            self.dt: The datatabel to be manipulated with\n
            self.colMap: A mapping between column names and the indcies. Can be seen by printing the class
                         or by setting help=True in the function removeColumns\n
        """
        if (file_path_csv is not None):
            self.__path = file_path_csv
            dt_t = pd.read_csv(file_path_csv,sep=r'\s*,\s*',engine='python')
            self.__normalized = False
            self.colMap = np.array([c[1] for c in enumerate(dt_t.columns)])
            self.dt = dt_t.to_numpy()
            self.__restoreColumn = np.array([])
            self.__restoreHeader = np.array([])
            self.dt_out = self.dt
            self.dt_in = self.dt

        elif (colMap is not None and dt is not None ):
            self.dt = dt
            self.colMap = colMap
            self.__restoreColumn = np.array([])
            self.__restoreHeader = np.array([])
            self.__normalized = False
            self.__path = "Mixed table from two different data sources"
            self.dt_out = self.dt
            self.dt_in = self.dt
        
        else:
            raise SyntaxError

    
    @classmethod
    def from2DH(cls,dh_1,dh_2,axis=0):
        """
            dh_1 is the first datahandler\n
            dh_2 is the second, which is appened to dh_1\n
            axis speicify if it should extend the rows or the columns. See Data_handler.append for more.
        """
        dh_1.append(dh_2,axis)
        return cls(colMap=dh_1.colMap,dt=dh_1.dt)

    
    @classmethod
    def dataAndHeader(cls,dt,header_l):
        """
            dt is the data\n
            header_l is a list of column names as they appear in order eg ["Mouse","Size","Speed"]\n
        """
        return cls(colMap=np.array(header_l),dt=dt)

    def __str__(self):
        return f"Datatabel has {self.dt.shape[0]} samples with {self.dt.shape[1]} columns\nThe column names that can be used as keys for the column mapping are as follows:\n{ self.colMap }"

    def __repr__(self):
        return f"Data analysis from path {self.__path}"

    def __saveToRestore(self,Matrix_data,array_colName):
        """
            Private function used to save data whenever this is removed from the tabel, if the user so chooses.
        """
        if self.__restoreColumn.size == 0:
            self.__restoreColumn = Matrix_data
        else:
            self.__restoreColumn = np.append(self.__restoreColumn,Matrix_data,axis=1)

        if self.__restoreHeader.size == 0:
            self.__restoreHeader = array_colName
        else:
            self.__restoreHeader = np.append(self.__restoreHeader,array_colName)

    
    def shape(self):
        return self.dt.shape
    
    def deleteSavedData(self):
        """
            This function removes all saved columns and header. This function is not reversable
        """
        self.__restoreColumn = np.array([])
        self.__restoreHeader = np.array([])



    def restoreSavedData(self):
        """
            This function restores the saved data at the end of the datatable. 
            Use the function moveColumn to rearange the order of the data.
        """
        if(self.__restoreColumn.size != 0):
            self.dt = np.append(self.dt,self.__restoreColumn,axis=1)
            self.colMap = np.append(self.colMap,self.__restoreHeader)
            self.deleteSavedData()
        else:
            print("There is no data to restore")
    def str_support(self):
        pass

    def setPath(self,name_path):
        self.__path = name_path

    
    def removeColumns(self,col_name_list,h=False,save = True):
        """
            Used to alter the datatabel by removing unwanted 
            columns.\n
            h: if set to True the column names of the datatable will be shown. Setting help till true
            will not alter the datatabel. (Default False)\n
            col_name_list: Take the name(s) of the coulmn(s) and remove all of them as stated in the list\n
            save: Default True - Saves the deletet columns to be restored later by calling .restoreData

        """
        if h:
            print("Available column name in the datatable:\n")
            for name in self.colMap:
                print(f"{name}")
                isinstance
        else:
            if len(col_name_list) > 0:
                indexis = [np.where(self.colMap == name)[0][0] for name in col_name_list]
            else:
                return

            if(save):
                self.__saveToRestore(self.dt[:,indexis],self.colMap[indexis])
            
            self.colMap = np.delete(self.colMap,indexis,0)
            self.dt = np.delete(self.dt,indexis,1)
    
    def saveToCsv(self,file_name,path = "..\\Data\\",floatPrecision = 3):
        with open(path+file_name+".csv", 'w+',encoding='utf-8') as file: 
            # Create header:
            for i in range(self.colMap.shape[0]):
                file.write("{0}".format(self.colMap[i]))
                if(i < (self.colMap.shape[0]-1)):
                    file.write(",")
                else:
                    file.write("\n")
                pass
            
            # Insert data:
            for row in self.dt:
                for i in range(row.shape[0]):
                    file.write(f"{row[i]:.{floatPrecision}f}")
                    if (i < (row.shape[0]-1)):
                        file.write(",")
                    else:
                        file.write("\n")    
      
    def moveColumn(self,new_indx,col_indx=-1,col_name = None):
        """
            moves an exisiting column.\n
            col_indx: The index of the columns which needs to be moved.\n
            col_name: An optional argument if set col_indx is ignored. Used to state the name of the column instead that needs to be moved.\n 
            new_indx: The new position of the column, if -1 is put in, it is set to the end

        """
        if (col_name != None):
            tmp_indx = np.where(self.colMap == col_name)[0]
            if(tmp_indx.size == 0):
                print(f"The name '{col_name}' can not be found, skips function")
                return
            else:
                tmp_indx = tmp_indx[0]
        else:
            tmp_indx = col_indx
        
        tmp_colName = self.colMap[tmp_indx]
        tmp_col = self.dt[:,tmp_indx].reshape(-1,1)
        
        #Remove the column from the datatable  
        self.colMap = np.delete(self.colMap,tmp_indx,0)
        self.dt = np.delete(self.dt,tmp_indx,1)

        #Set it in again in the new postion
        if(new_indx == -1 or new_indx == self.dt.shape[1]):
            self.colMap = np.append(self.colMap,tmp_colName)
            self.dt = np.append(self.dt,tmp_col,axis=1)
        else:    
            self.colMap = np.insert(self.colMap,new_indx,tmp_colName)
            self.dt = np.insert(self.dt,new_indx,tmp_col,axis=1)

        pass
        
    
    def mult(self,x):
        self.dt *= x
        
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


    def append(self,data_handler,axis=0, new_file_path_name = ""):
        """
            This function extend the data_handler with more data from another data_handler.\n
            If axis = 0, then it extends the rows, the column name and position of the names is expected to be the same. \n
            If axis = 1, the column will be extended accordingly \n
        """
        if self.__restoreColumn.size != 0 or self.__restoreHeader != 0:
            print("There is some saved data which can be restored. Either restore this or deletet the buffers in order to use this function")
            return

        if (axis==0):
            if (data_handler.dt.shape[1] != self.dt.shape[1]):
                print("The number of columns doesn't match")
                print(f"Shape {data_handler.dt.shape}")
                print(f"Shape {self.dt.shape}")
            else:
                self.dt = np.append(self.dt,data_handler.dt,axis=axis)
                print("Succesfully extended number of rows in self.dt")
        else:
            if (data_handler.dt.shape[0] != self.dt.shape[0]):
                print("The number of rows has to be teh same")
                print(f"Shape {data_handler.dt.shape}")
                print(f"Shape {self.dt.shape}")
            else:
                readyForAppend = True
                for col_name in data_handler.colMap:
                    if (np.where(self.colMap == col_name)[0].size != 0):
                        print(f"The column name {col_name} already excist in the datatable rename this column.")
                        readyForAppend = False
                if readyForAppend:
                    self.dt = np.append(self.dt,data_handler.dt,axis=axis)
                    self.colMap = np.append(self.colMap,data_handler.colMap)
                    print("Succesfully extended number of columns in self.dt, and added names to self.colMap")
    
    def renameCoulmn(self,new_name,col_indx=-1,col_name = None):
        """
            Renames an exisiting column.\n
            col_indx: The index of the columns which needs to be renamed.\n
            col_name: An optional argument if set col_indx is ignored.\n 
            new_name: The new name of the columns

        """
        if (col_name != None):
            self.colMap[np.where(self.colMap == col_name)] = new_name
        else:
            self.colMap[col_indx] = new_name


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
        for i in range(self.dt.shape[1]):
            yhat = iso.fit_predict(self.dt[:,i].reshape(-1,1))
            mask = yhat != -1
            self.dt = self.dt[mask,:]

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
                pca = PCA(n_components=1)
                for i in range(min_comp,self.dt_out.shape[1]+1):
                    pca = PCA(n_components=i)
                    pca.fit(self.dt_out)
                    if(sum(pca.explained_variance_ratio_) >= explain_var):
                        return pca
                return pca
        else:
            pca = PCA(n_components=1)
            for i in range(min_comp,self.dt.shape[1]+1):
                pca = PCA(n_components=i)
                pca.fit(self.dt)
                if(sum(pca.explained_variance_ratio_) >= explain_var):
                    return pca
            return pca






class DEA(object):

    def __init__(self, DH, input_indx_split):
        """
            DH: A datahandler with all the preprocessing done.\n
            input_indx_split: an int indicating where the input stop\n
            self.input: Range from [0:input_indx_split[\n
            self.output: Range from [input_indx_split:-1]\n

        """

        # supplied data
        self.inputs = DH.dt[:80,:input_indx_split]
        self.outputs = DH.dt[:80,input_indx_split:]

        # parameters
        self.n = self.inputs.shape[0]
        self.m = self.inputs.shape[1]
        self.r = self.outputs.shape[1]

        # iterators
        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        #
        # lower_bound = np.zeros(8) - np.inf
        # upper_bound = np.zeros(8)
        # self.bounds = [(l,h) for l,h in zip(lower_bound,upper_bound)]
        # print(self.bounds)

        # result arrays
        self.output_w = np.zeros((self.r, 1), dtype=np.float)  # output weights
        self.input_w = np.zeros((self.m, 1), dtype=np.float)  # input weights
        self.lambdas = np.zeros((self.n, 1), dtype=np.float)  # unit efficiencies
        self.efficiency = np.zeros_like(self.lambdas)  # thetas
   
   
    
    
    def __efficiency(self, unit):
        # compute efficiency
        denominator = np.dot(self.inputs, self.input_w)
        numerator = np.dot(self.outputs, self.output_w)

        return (numerator/denominator)[unit]
    def __target(self, x, unit):

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        denominator = np.dot(self.inputs[unit], in_w)
        numerator = np.dot(self.outputs[unit], out_w)
        return numerator/denominator

    def __constraints(self, x, unit):

        in_w, out_w, lambdas = x[:self.m], x[self.m:(self.m+self.r)], x[(self.m+self.r):]  # unroll the weights
        constr = []  # init the constraint array

        # for each input, lambdas with inputs
        for input in self.input_:
            t = self.__target(x, unit)
            lhs = np.dot(self.inputs[:, input], lambdas)
            cons = t*self.inputs[unit, input] - lhs
            constr.append(cons)

        # for each output, lambdas with outputs
        for output in self.output_:
            lhs = np.dot(self.outputs[:, output], lambdas)
            cons = lhs - self.outputs[unit, output]
            constr.append(cons)

        # for each unit
        for u in self.unit_:
            constr.append(lambdas[u])
        return np.array(constr)

    def __optimize(self):

        d0 = self.m + self.r + self.n
        # iterate over units
        for unit in self.unit_:
            # weights
            x0 = np.random.rand(d0) - 0.5
            x0 = fmin_slsqp(self.__target, x0, f_ieqcons=self.__constraints, args=(unit,),iprint=False)
            # unroll weights
            self.input_w, self.output_w, self.lambdas = x0[:self.m], x0[self.m:(self.m+self.r)], x0[(self.m+self.r):]
            self.efficiency[unit] = self.__efficiency(unit)

    def fit(self):
        self.__optimize()  # optimize
        return self.efficiency




# ----------- Stand alone function ---------------- #
def progressBar(current, total, barLength = 20):
    #Lend from: https://stackoverflow.com/questions/6169217/replace-console-output-in-python
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('       Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')









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






