
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
reload(utilize)
reload(multiVariant)
reload(singleVariant)
reload(multiHistogramBase)
reload(CopulaBase)



startTime=time.time()

attribute_names=np.array(["phi_grav","particle_mass_density","zmom","ymom"])
incremental_number=300
all_ensamble_data=np.empty([0,incremental_number,64,64,64])

for name in attribute_names:
    data=utilize.readFiles(name,incremental_number)
    data=data.reshape(1,incremental_number,64,64,64)
    all_ensamble_data=np.append(all_ensamble_data,data,axis=0)

#print(all_ensamble_data.shape)
#print(all_ensamble_data[0].shape)
covBlockSize=5
dataBlockSize=5
binsNumber=128
sizeZ=60
sizeY=60
sizeX=60
minMaxBlockSize=2
isMinMax=False
print("new")
print("start fit model")
with tqdm(total=4, desc="Model fitting") as pbar:
    #oursModel=multiVariant.multiDistCopula3D(all_ensamble_data,dataBlockSize,covBlockSize,binsNumber,[sizeZ,sizeY,sizeX],minMaxBlockSize,isMinMax)
    oursModel=multiVariant.multiDistCopula3D.load(f"Nyx_{attribute_names.shape[0]}varaibles_{incremental_number}members_128Bins_dBlock5_cBlock5_new")
    #conditions=np.array([[0,1e5],[3e10,5e10]])
    #oursModel.fit()
    print("ours complete fit")
    pbar.update(1)
    copulaBlockSize=2
    copulaBaseModel=CopulaBase.multiVariantCopulaBase(all_ensamble_data,copulaBlockSize)
    copulaBaseModel.fit()
    print("copula complete fit")
    pbar.update(1)
    multiHistBlockSize=2
    multiHistModel=multiHistogramBase.multiHistogramModel(all_ensamble_data,blockSize=multiHistBlockSize,binsNumber=binsNumber)
    multiHistModel.fit()
    print("multi-hist complete fit")
    pbar.update(1)
    gtModel=multiHistogramBase.multiHistogramModel(all_ensamble_data,blockSize=1,binsNumber=binsNumber)
    gtModel.fit()

    multiBinEdges=gtModel.blocks[0].binEdges

    print("complete fit")
    pbar.update(1)

oursError=cp.zeros([sizeZ,sizeY,sizeX],dtype=cp.float32)
copulaError=cp.zeros([sizeZ,sizeY,sizeX],dtype=cp.float32)
mtError=cp.zeros([sizeZ,sizeY,sizeX],dtype=cp.float32)


multiBinEdges=cp.asarray(multiBinEdges,dtype=cp.float32)


with tqdm(total=sizeZ*sizeY*sizeX, desc="總進度") as pbar:
    for idx in range(sizeZ * sizeY * sizeX):
        
        z = idx // (sizeY * sizeX)
        y = (idx // sizeX) % sizeY
        x = idx % sizeX        
        ### GroundTruth ###

        gtMultiHist=gtModel.getHistByPosCp(z,y,x)
        gtMultiHist=gtMultiHist/cp.sum(gtMultiHist)

        ### ours method ###

        oursSamples=oursModel.sampleByPos(z,y,x)
        oursSamples=cp.asarray(oursSamples,dtype=cp.float32)
        oursMultiHist,_=cp.histogramdd(oursSamples,bins=multiBinEdges)
        oursMultiHist=oursMultiHist/cp.sum(oursMultiHist)

        #rmse=np.sqrt(np.mean((gtMultiHist-oursMultiHist)**2))
        rmse = cp.sqrt(cp.mean((gtMultiHist - oursMultiHist) ** 2))
        oursError[z,y,x]=rmse
        #oursError.append(rmse)

       
        ### copula Base ###
    
        copulaSamples=copulaBaseModel.sampleByPos(z,y,x)
        copulaSamples=cp.asarray(copulaSamples,dtype=cp.float32)
        copulaMultiHist,_=cp.histogramdd(copulaSamples,bins=multiBinEdges)
        copulaMultiHist=copulaMultiHist/cp.sum(copulaMultiHist)
      
        rmse = cp.sqrt(cp.mean((gtMultiHist - copulaMultiHist) ** 2))
        #rmse=np.sqrt(np.mean((gtMultiHist-copulaMultiHist)**2))
        copulaError[z,y,x]=rmse
        #copulaError.append(rmse)
        

        ### multiHist ###

        ProcessTime=time.time()
        ProcessHistTime=time.time()
        mtMultiHist=multiHistModel.getHistByPosCp(z,y,x)
        mtMultiHist=mtMultiHist/cp.sum(mtMultiHist)
        processHistTimeEnd=time.time()
        #print(f"multi hist hist 執行時間:{processHistTimeEnd-ProcessHistTime}")
        processRmseTime=time.time()
        #rmse=root_mean_squared_error(gtMultiHist.flatten(),mtMultiHist.flatten())
        #rmse=np.sqrt(np.mean((gtMultiHist-mtMultiHist)**2))
        rmse = cp.sqrt(cp.mean((gtMultiHist - mtMultiHist) ** 2))
        processRmseTimeEnd=time.time()
        #print(f"multi hist RMSE 執行時間:{processRmseTimeEnd-processRmseTime}")
        mtError[z,y,x]=rmse
        #mtError.append(rmse)

        ProcessTimeEnd=time.time()
        #print(f"multi hist 執行時間:{ProcessTimeEnd-ProcessTime}")
        
        pbar.update(1)
    

#oursError=np.array(oursError)
#copulaError=np.array(copulaError)
#mtError=np.array(mtError)

oursError=oursError.get()
copulaError=copulaError.get()
mtError=mtError.get()

oursError=oursError.mean()
copulaError=copulaError.mean()
mtError=mtError.mean()

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Outputfilename = f"output_{timestamp}.txt"
end_Time=time.time()

with open(Outputfilename , "w", encoding="utf-8") as f:
    print(f"ours error:{oursError.mean()}",file=f)
    print(f"copula error:{copulaError.mean()}",file=f)
    print(f"mt error: {mtError.mean()}",file=f)
    print(f"執行時間:{end_Time-startTime}",file=f)

