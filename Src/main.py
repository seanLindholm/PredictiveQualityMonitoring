from Helper import Data_handler, prettyprint as pp


#The path to the data used
data_path = "..\\Data"
approved_file_name = data_path + "\\Approved_w_glu.csv"
failed_file_name = data_path + "\\Failed_w_glu.csv"
approved_file_Transform_name = data_path + "\\approved_trimed_transform.csv"
failed_file_Transform_name = data_path + "\\failed_trimed_transform.csv"
eff_failed_name = data_path + "\\eff_failed.csv"
eff_approved_name = data_path + "\\eff_approved.csv"
eff_mixed_name = data_path + "\\eff_mixed.csv"


def main():
    dh = Data_handler(approved_file_name)
    dh_2 = Data_handler(failed_file_name)

    dh.dt[:,3] = 1
    dh_2.dt[:,3] = 0
    print(dh)
    print()
    dh.moveColumn(-1,col_name='Class')
    print(dh)
    print()
    dh.removeColumns(['Class','YM'])
    print(dh)
    print()
    dh.restoreSavedData()
    print(dh)
    print()
    dh.append(dh_2)
    # dh.restoreSavedData()
    # print(dh)
    # print()
    # dh.moveColumn(2,col_name='YM')
    # print(dh)
    # print()
    # pass

if __name__ == "__main__":
    main()