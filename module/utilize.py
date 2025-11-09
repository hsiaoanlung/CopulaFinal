import numpy as np
def readFile(attribute_name,ensamble_index):
    f=open("../nyx_data/"+attribute_name+"/"+attribute_name+str(ensamble_index+100)+".bin")
    temp=np.fromfile(f,dtype=np.float32,count=-1)
    temp=temp.reshape(64,64,64)
    f.close()

    return temp

def readFiles(attribute_name,ensamble_member):
    name=[attribute_name for x in range(ensamble_member)]
    index=np.arange(ensamble_member)
    data=list(map(readFile,name,index))
    data=np.array(data)
    return data


def readRedSeaFile(attribute_name,filePath=None):
    if filePath==None:
        f=open(f"../red_data/{attribute_name}_240_30_180_180.bin")
    else:
        f=open(filePath+f"{attribute_name}_240_30_180_180.bin")
    temp=np.fromfile(f,dtype=np.float32,count=-1)
    temp=temp.reshape(240,30,180,180)
    f.close()

    return temp