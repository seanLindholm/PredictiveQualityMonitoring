import os
import pandas as pd
from datetime import datetime
from constants_ import outlierRemoval

def main():
    

    #csv_file = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\Failed_w_glu_Transform_ext.csv"
    #csv_file = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\Poor_func_failed.csv"
    csv_file = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\Poor_func_failed.csv"

    out_csv_file = "failed_ext.csv"
    save_dir = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\bcr_files\\" 
    
    #First remove outliers
    #outlierRemoval(csv_file,'Tid efter start [timer]')
    findAndCopyBCRFiles(csv_file,save_dir,out_csv_file,True)
    print("Next file!")
    
    #csv_file = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Data\\Approved_w_glu_Transform_ext.csv"
    csv_file = "C:\\Users\\swang\\Desktop\\Sean\\Speciale\\PredictiveQualityMonitoring\\Data\\Poor_func_approved.csv"
    out_csv_file = "approved_ext.csv"
    #First remove outliers
    #outlierRemoval(csv_file,'Tid efter start [timer]')
    findAndCopyBCRFiles(csv_file,save_dir,out_csv_file,False)


def getData(data_path):
    return pd.read_csv(data_path,sep=r'\s*,\s*',engine='python',encoding='latin_1')


def findAndCopyBCRFiles(csv_file,save_dir,output_file,IsFailed,save_counter=100):
    date_csv = "C:\\Users\\SEALI\\OneDrive - Danaher\\Desktop\\Seans_opgaver\\Speciale\\PredictiveQualityMonitoring\\Src\\date_stoppers.csv"

    path_dir = "Y:\\4746 - Processdata\\7_3DSCAN\\ODIN\\"
    w_scanners= ["U66794_ID1","U54497_ID2","U66794_ID3","U66794_ID5","U66794_ID6"]
    dt_date = getData(date_csv)
    stop_date = dt_date['failed_bcr_date'][0] if IsFailed else dt_date['approved_bcr_data'][0]
    new_stop = stop_date
    stop_date = datetime.strptime(stop_date,'%d-%m-%Y %H:%M')
    dt = getData(csv_file)
    counter = 0
    res_counter = 0
    file_counter = 0
    foundNewStopDate = False
    for col in dt.columns: 
        first_col = col
        break 

    for index,r in dt.iterrows():
        dict_ = f"{r['PartNumber']}-{r['RunNumber']}-N00{r['BoardNumber']}"

        # First locate path to dict
        res = ""
        for scanner in w_scanners:
            res = findDictMatch(dict_,path_dir + scanner)
            if(res != ""):
                break
        curr_time = datetime.strptime(r['Timestamp'],'%d-%m-%Y %H:%M')
        # 20 files in a row with no data. Expect there to be none
        if(res_counter == 50 or curr_time <= stop_date):
            break
        
        if (res == ""): print("Didn't find anyting, aboard!"); res_counter+=1;continue
        res_counter = 0

        # Now find the three .bcr files and store them on dekstop
        # In a folder named prod-run-board (later also put into the data itself)

        # Construct the three files we are looking for:
        f1 = f"Glu_Kavitet_r1_a{r['ArrayNumber']}.bcr" if r['ArrayNumber'] <= 8 else f"Glu_Kavitet_r2_a{r['ArrayNumber']-8}.bcr"
        f2 = f"Glu_CA_r1_a{r['ArrayNumber']}.bcr" if r['ArrayNumber'] <= 8 else f"Glu_CA_r2_a{r['ArrayNumber']-8}.bcr"
        f3 = f"Glu_YM_r1_a{r['ArrayNumber']}.bcr" if r['ArrayNumber'] <= 8 else f"Glu_YM_r2_a{r['ArrayNumber']-8}.bcr"
        
        files = [f1,f2,f3]
        res_files=[]
        for file_ in files:
            res_files.append(findDictMatch(file_,res))


        #Now where all files have been found (hopefully) copy them to another location
        save_folder = f"{dict_}-A{r['ArrayNumber']}-{r['Class']}"
        save_path = save_dir+f"{dict_}-A{r['ArrayNumber']}-{r['Class']}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        names = ["RAW","CA","YM"]
        dt.loc[index,first_col] = save_folder            

        file_counter = 0
        for file_path,name in zip(res_files,names):
            os.system(f'copy "{file_path}" "{save_path}\\{name}.bcr"')
            file_counter += 1
        
        if(file_counter == 3):
            print(f"Current timestamp: {curr_time}, stop by {stop_date}, new stop: {new_stop}")
            if(not foundNewStopDate):
                foundNewStopDate = True
                new_stop = r['Timestamp']
            dt.loc[index,first_col] = save_folder  
        

        counter+=1
        if(counter == save_counter):
            print("Saved file so far")
            dt.to_csv(output_file,index=False,encoding='latin-1')
            counter = 0
    
    #Save the csv files one last time
    dt.to_csv(output_file,index=False,encoding='latin-1')
    if IsFailed:
        dt_date['failed_bcr_date'] = new_stop
    else:
        dt_date['approved_bcr_data'] = new_stop
    dt_date.to_csv(date_csv,index=False,encoding='latin-1')






def listDicts(path):
        for dict_ in os.listdir(path):
            yield dict_

def findDictMatch(dict_,path):
    for d in listDicts(path):
        if d == dict_:
            return path + "\\" + dict_
    return ""



if __name__ == "__main__":
    main()