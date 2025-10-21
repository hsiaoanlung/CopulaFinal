from importlib import reload ,import_module
import module.utilize as utilize
import module.multiVariant as multiVariant
import module.singleVariant as singleVariant
import numpy as np
from numba import njit,jit, float32
from tqdm import tqdm
reload(utilize)
reload(multiVariant)
reload(singleVariant)

if __name__ == '__main__':
    attribute_names=np.array(["phi_grav","particle_mass_density","zmom","ymom","xmom"])
    incremental_number=300
    all_ensamble_data=np.empty([0,incremental_number,64,64,64])

    for name in attribute_names:
        data=utilize.readFiles(name,incremental_number)
        data=data.reshape(1,incremental_number,64,64,64)
        all_ensamble_data=np.append(all_ensamble_data,data,axis=0)

    #print(all_ensamble_data.shape)
    #print(all_ensamble_data[0].shape)
    covBlockSize=2
    dataBlockSize=5
    binsNumber=128
    sizeZ=60
    sizeY=60
    sizeX=60
    minMaxBlockSize=2
    isMinMax=False

    with tqdm(total=4, desc="總進度") as pbar:
        for i in range(2,6):
            data=all_ensamble_data[0:i, :, :, :, :]
            
            copulaModel=multiVariant.multiDistCopula3D(data,dataBlockSize,covBlockSize,binsNumber,[sizeZ,sizeY,sizeX],minMaxBlockSize,isMinMax)
            copulaModel.fit()
            copulaModel.saveInfoToFile(f"Nyx_{i}varaibles_{incremental_number}members_{binsNumber}Bins_dBlock{dataBlockSize}_cBlock{covBlockSize}_new")
            copulaModel.saveModel(f"Nyx_{i}varaibles_{incremental_number}members_{binsNumber}Bins_dBlock{dataBlockSize}_cBlock{covBlockSize}_new")

            pbar.update(1)
