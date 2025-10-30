#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

reload(utilize)
reload(multiVariant)
reload(singleVariant)
reload(multiHistogramBase)
reload(CopulaBase)

startTime=time.time()

attribute_names=np.array(["phi_grav","particle_mass_density","xmom"])
incremental_number=400
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

print("start fit model")
with tqdm(total=4, desc="Model fitting") as pbar:
    oursModel=multiVariant.multiDistCopula3D(all_ensamble_data,dataBlockSize,covBlockSize,binsNumber,[sizeZ,sizeY,sizeX],minMaxBlockSize,isMinMax)

    conditions=np.array([[0,1e5],[3e10,5e10]])
    oursModel.fit()
    print("ours complete fit")
    pbar.update(1)
    copulaBlockSize=2
    #oriData=all_ensamble_data[:,:,0:sizeZ, 0:sizeY, 0:sizeX]
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

oursError=[]
copulaError=[]
mtError=[]

with tqdm(total=sizeZ*sizeY*sizeX, desc="總進度") as pbar:
    for z in range(sizeZ):
        for y in range(sizeY):
            for x in range(sizeX):

                ### GroundTruth ###
                gtSamples=gtModel.sampleByPos(z,y,x)
                gtMultiHist,_=np.histogramdd(gtSamples,bins=multiBinEdges)
                gtMultiHist=gtMultiHist/np.sum(gtMultiHist)

                ### ours method ###
                oursSamples=oursModel.sampleByPos(z,y,x)
                oursMultiHist,_=np.histogramdd(oursSamples,bins=multiBinEdges)
                oursMultiHist=oursMultiHist/np.sum(oursMultiHist)

                rmse=np.sqrt(np.mean((gtMultiHist-oursMultiHist)**2))
                oursError.append(rmse)
                ### copula Base ###

                copulaSamples=copulaBaseModel.sampleByPos(z,y,x)
                copulaMultiHist,_=np.histogramdd(copulaSamples,bins=multiBinEdges)
                copulaMultiHist=copulaMultiHist/np.sum(copulaMultiHist)

                rmse=np.sqrt(np.mean((gtMultiHist-copulaMultiHist)**2))
                copulaError.append(rmse)
                ### multiHist ###

                mtSamples=multiHistModel.sampleByPos(z,y,x)
                mtMultiHist,_=np.histogramdd(mtSamples,bins=multiBinEdges)
                mtMultiHist=mtMultiHist/np.sum(mtMultiHist)

                rmse=np.sqrt(np.mean((gtMultiHist-mtMultiHist)**2))
                mtError.append(rmse)

                pbar.update(1)


oursError=np.array(oursError)
copulaError=np.array(copulaError)
mtError=np.array(mtError)


from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
Outputfilename = f"output_{timestamp}.txt"
end_Time=time.time()

with open(Outputfilename , "w", encoding="utf-8") as f:
    print(f"ours error:{oursError.mean()}",file=f)
    print(f"copula error:{copulaError.mean()}",file=f)
    print(f"mt error: {mtError.mean()}",file=f)
    print(f"執行時間:{end_Time-startTime}",file=f)

