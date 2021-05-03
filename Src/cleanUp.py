import os
from constants_ import path
def removeDublicates(list_):
    del_list = []    
    for i in range(len(list_)):
        for j in range((i+1),len(list_)):
            if list_[i] == list_[j]:
                del_list.append(list_[i])
    return del_list
  
def removeFolders(list_):
    for l in list_:
        os.system(f"rmdir /s {path+l}-Approved")
        os.system(f"rmdir /s {path+l}-Failed")

def removeFoldersWithOddImageCount(img_count):
    c = input(f"Are you sure that {img_count} is the number of files that should be in each folder [y/n] ").lower()
    if (c=="y"):
        for l in os.listdir(path):
            len_ = len(os.listdir(path+l))
            if img_count != len_:
                os.system(f"rmdir /s {path+l}")
    else:
        return
def main():
    list_ = [ l.replace("-Approved", '').replace("-Failed", "") for l in os.listdir(path) ]
    del_ = removeDublicates(list_)
    removeFolders(del_)
    # REMEMBER TO MAKE SURE THAT THE NUMBER IS CORRECT!
    removeFoldersWithOddImageCount(18)
    

if __name__ == "__main__":
    main()