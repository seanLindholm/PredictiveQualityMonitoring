import pandas as pd
import numpy as np
from constants import *
"""
 This script works entirely on the transformed data, and will be updated as 
 new data is extracted from the differetn process

 Rev 1.0: pandas used to extract the information from failed and approved to dea and back again.
"""


def main():
    #makeDEAReady("dea_mixed.csv")
    # Has to be done in stages since the dea calcuations are done in matlab
    appendDEAData()

def appendDEAData():
    df_fail = getData(failed_NoNaN); df_approved = getData(approved_NoNaN)
    dea_eff = getData(data_path + "eff_mixed.csv")
    f_len = len(df_fail)
    a_len = len(df_approved)
    dea_len = len(dea_eff)
    df_f_index = np.arange(f_len)
    df_a_index = np.arange(f_len,dea_len)
    df_fail.insert(len(df_fail.columns),"DEA",dea_eff["dea_eff"][df_f_index].tolist(),True)
    df_approved.insert(len(df_approved.columns),"DEA",dea_eff["dea_eff"][df_a_index].tolist(),True)
    saveDF(df_fail,"failed_withDEAScore")
    saveDF(df_approved,"approved_withDEAScore")


def makeDEAReady(save_name):
    # Get the df with the columns of interest
    df = getData(failed_NoNaN)[function_test_col_transformed]
    df_2 = getData(approved_NoNaN)[function_test_col_transformed]
    df = df.append(df_2)
    saveDF(df,save_name)

def concatDEA(dea_file,file,save_name):
    pass

if __name__ == "__main__":
    main()