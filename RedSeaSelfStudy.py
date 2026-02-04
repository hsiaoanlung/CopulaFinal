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
from sklearn.metrics import root_mean_squared_error
import cupy as cp
import module.multiHistogramSparse as multiHistogramSparse
import module.ananlysisFuncion as ananlysisFunction
reload(utilize)
reload(multiVariant)
reload(singleVariant)
reload(multiHistogramBase)
reload(CopulaBase)
reload(multiHistogramSparse)
reload(ananlysisFunction)


startTime=time.time()

#attribute_names=np.array(["SALT","TEMP",])
attribute_names=np.array(["SALT","TEMP","U","V","W"])

all_ensamble_data=np.empty([0,240,30,180,180])
incremental_number=240

for name in attribute_names:
    data=utilize.readRedSeaFile(name)
    data=data.reshape(1,240,30,180,180)
    all_ensamble_data=np.append(all_ensamble_data,data,axis=0)


covBlockSize=5
dataBlockSize=6
binsNumber=128
sizeZ=30
sizeY=180
sizeX=180
minMaxBlockSize=2
isMinMax=False
all_ensamble_data=all_ensamble_data[:,:, 0:sizeZ, 0:sizeY, 0:sizeX]

for i in range(2,6):
    data=all_ensamble_data[0:i, :, :, :, :]
    print("start fit model")
    with tqdm(total=2, desc="Model fitting") as pbar:
    
        oursModel=multiVariant.multiDistCopula3D.load(f"RedSea_{i}varaibles_{incremental_number}members_{binsNumber}Bins_dBlock{dataBlockSize}_cBlock5")
        
        print("ours complete fit")
        pbar.update(1)

        gtModel=multiHistogramSparse.multiHistogramSpaseModel(data,blockSize=1,binsNumber=binsNumber)
        gtModel.fit()

        multiBinEdges=gtModel.vBinEdges

        print("complete fit")
        pbar.update(1)

    oursError=[]
    


    #multiBinEdges=cp.asarray(multiBinEdges,dtype=cp.float32)


    with tqdm(total=sizeZ*sizeY*sizeX, desc="總進度") as pbar:
        for idx in range(sizeZ * sizeY * sizeX):
            
            z = idx // (sizeY * sizeX)
            y = (idx // sizeX) % sizeY
            x = idx % sizeX        
            ### GroundTruth ###

            gtMultiHistModel=gtModel.getHistByPos(z,y,x)
            coords_gt, vals_gt = gtMultiHistModel.to_coo()
            ### ours method ###

            oursSamples=oursModel.sampleByPos(z,y,x)
            oursHistModel=multiHistogramSparse.SparseMultiHistogramBlock(bin_edges=multiBinEdges)
            oursHistModel.add_samples(oursSamples)
            oursHistModel.normalize()

            coords_target,vals_target=oursHistModel.to_coo()
            emd=ananlysisFunction.emd_sparse(coords_gt,vals_gt,coords_target,vals_target)
            oursError.append(emd)
            
            pbar.update(1)
        

    oursError=np.array(oursError)
    oursError=oursError.mean()

    with open(f"RedSeaSelfEMD_DBlock{dataBlockSize}_Bin{binsNumber}.txt", "a", encoding="utf-8") as f:  # 使用 "a" 表示 append
        f.write(f"Variable:{i} ,binNumber:{binsNumber}, oursError:{oursError}\n")  # 每次寫入並換行