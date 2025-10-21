from importlib import reload ,import_module
import module.utilize as utilize
import module.multiVariant as multiVariant
import module.singleVariant as singleVariant
import module.multiHistogramBase as multiHistogramBase
import numpy as np
from numba import njit,jit, float32
import module.singleVariantCopulaBase as CopulaBase
from tqdm import tqdm
import time
from multiprocessing import Pool
reload(utilize)
reload(multiVariant)
reload(singleVariant)
reload(multiHistogramBase)
reload(CopulaBase)

oursModel=None
sizeZ=10
sizeY=10
sizeX=10

def compute_Error(args):
    idx, oursModel=args
    z = idx // (sizeY * sizeX)
    y = (idx // sizeX) % sizeY
    x = idx % sizeX
    samples=oursModel.sampleByPos(z,y,x)
    mean=samples.mean()
    return mean

if __name__ == '__main__':
    startTime=time.time()

    attribute_names=np.array(["phi_grav","particle_mass_density","xmom"])
    incremental_number=300
    all_ensamble_data=np.empty([0,incremental_number,64,64,64])
    """
    for name in attribute_names:
        data=utilize.readFiles(name,incremental_number)
        data=data.reshape(1,incremental_number,64,64,64)
        all_ensamble_data=np.append(all_ensamble_data,data,axis=0)
    """
    oursModel=multiVariant.multiDistCopula3D.load(f"Nyx_{attribute_names.shape[0]}varaibles_{incremental_number}members_128Bins_dBlock5_cBlock5_new")
    
    

    args_list = [(i, oursModel)
                for i in range(sizeZ*sizeY*sizeX)]

    with Pool(4) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(compute_Error, args_list),  # 或 pool.imap_unordered
                total=len(args_list),
                desc="Processing in parallel"
            )
        )
