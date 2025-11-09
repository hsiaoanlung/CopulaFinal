from importlib import reload ,import_module
import module.utilize as utilize
import module.singleVariant as singleVariant
import module.multiVariant as multiVariant
import numpy as np
from numba import njit,jit, float32
from tqdm import tqdm
reload(utilize)
reload(multiVariant)
reload(singleVariant)

attribute_names=np.array(["SALT","TEMP","U","V","W"])

all_ensamble_data=np.empty([0,240,30,180,180])
for name in attribute_names:
    data=utilize.readRedSeaFile(name)
    data=data.reshape(1,240,30,180,180)
    all_ensamble_data=np.append(all_ensamble_data,data,axis=0)

incremental_number=240

conditions=np.array([[36,40],[26,30]])

covBlockSize=5
dataBlockSize=5
binsNumber=128
sizeZ=30
sizeY=180
sizeX=180
minMaxBlockSize=2
isMinMax=False

with tqdm(total=4, desc="總進度") as pbar:
    for i in range(2,6):
        data=all_ensamble_data[0:i, :, :, :, :]

        copulaModel=multiVariant.multiDistCopula3D(data,dataBlockSize,covBlockSize,binsNumber,[sizeZ,sizeY,sizeX],minMaxBlockSize,isMinMax)
        copulaModel.fit()
        copulaModel.saveInfoToFile(f"RedSea_{i}varaibles_{incremental_number}members_{binsNumber}Bins_dBlock{dataBlockSize}_cBlock{covBlockSize}")
        copulaModel.saveModel(f"RedSea_{i}varaibles_{incremental_number}members_{binsNumber}Bins_dBlock{dataBlockSize}_cBlock{covBlockSize}")

        pbar.update(1)