import pandas as pd

#path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
#file_path = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\"

path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\"
file_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Src\\"
data_path = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\"
failed = file_path + "failed_ext.csv"
failed_NoNaN = file_path + "failed_NoNaN.csv"

approved = file_path + "approved_ext.csv"
approved_NoNaN = file_path + "approved_NoNaN.csv"


function_test_col_transformed = ["Tid efter start [timer]","2/1 mM Glu/Lac [mM]","1 mM H2O2 [mM]","40/25 mM glu/lac høj O2",
                                 "Sensitivity [pA/µM]","t on 10/5 mM glu/lac [s]","Lav O2 - Høj O2"]


def saveDF(df,name):
    df.to_csv(name,index=False)

def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',na_values='')
