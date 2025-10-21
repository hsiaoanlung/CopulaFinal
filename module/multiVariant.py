import numpy as np
from numpy.typing import NDArray
from typing import List
from numba import njit, float32
from scipy.stats import multivariate_normal, norm
import module.singleVariant as singleVariant
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from multiprocessing import Pool
import os
import pickle
class blockCov():
    def __init__(self,oriData:NDArray[np.float32],dim):
        self.oriData=oriData

        self.mean=np.zeros(dim)
        self.counts=0
        self.M2=np.zeros((dim, dim)) #用來累積差平方和
        self.M2_diag = np.zeros(dim) 
    
    def fit(self):
        for i in range(self.oriData.shape[0]):
            self.update(self.oriData[i])
        
        self.oriData=[]
        
    def update (self, x:np.ndarray):
        self.counts+=1
        delta = x-self.mean
        self.mean+=delta/self.counts
        delta2=x-self.mean
        self.M2+=np.outer(delta,delta2)

        # 更新變異數對角元素
        self.M2_diag += delta * delta2
    
    def getMean(self):
        return self.mean
    
    def getCovMatrix(self):
        if self.counts <2:
            return np.zeros_like(self.M2)
        
        return self.M2 / (self.counts-1)
    
    ## pearson correlation ###
    def getCorrMatrix(self):
        if self.counts < 2:
            return np.zeros_like(self.M2)

        std = np.sqrt(self.M2_diag / (self.counts - 1))
        std_matrix = np.outer(std, std)
        cov = self.getCovMatrix()

        # 避免除以0
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.divide(cov, std_matrix, out=np.zeros_like(cov), where=std_matrix != 0)
        return corr

class multiCov():

    def __init__(self,oriData:NDArray[np.float32],blockSize):
        self.oriData=oriData
        self.blockSize=blockSize

        self.dim = oriData.shape[0]
        self.incrementalNumber = oriData.shape[1]

        self.sizeZ=self.oriData.shape[2]
        self.sizeY=self.oriData.shape[3]
        self.sizeX=self.oriData.shape[4]

        self.blocks:List[blockCov]=[]

    def fit(self):

        for z in range(0,self.sizeZ,self.blockSize):
            for y in range(0, self.sizeY,self.blockSize):
                for x in range(0, self.sizeX,self.blockSize):
                    
                    data=[]
                    for vIndex in range(self.dim):
                        v=self.oriData[vIndex,:, z : z+self.blockSize, y: y+self.blockSize, x: x+self.blockSize]
                        v=v.flatten()
                        data.append(v)
                    
                    data=np.array(data).astype(np.float32)
                    data=np.stack(data,axis=1)
                    block=blockCov(data,self.dim)
                    self.blocks.append(block)
        
        for i in range(len(self.blocks)):
            self.blocks[i].fit()
        
        self.oriData=[]
    
    def getPointCov(self,z,y,x):
        blockWidth=self.sizeZ//self.blockSize
        blockZ, blockY, blockX = z//self.blockSize, y//self.blockSize, x//self.blockSize
        blockIdx = blockZ * (blockWidth**2) + blockY * blockWidth + blockX

        covMatrix = self.blocks[blockIdx].getCovMatrix()
        return covMatrix
    
    def getPointCorr(self,z,y,x):

        blockWidthZ=self.sizeZ//self.blockSize
        blockWidthY=self.sizeY//self.blockSize
        blockWidthX=self.sizeX//self.blockSize
        blockZ, blockY, blockX = z//self.blockSize, y//self.blockSize, x//self.blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX

        corrMatrix = self.blocks[blockIdx].getCorrMatrix()
        return corrMatrix

class multiDistCopula3D():
    def __init__(self,oriData:NDArray[np.float32],dataBlockSize,covBlockSize,binsNumber,allDataSize=[15,15,15],minMaxBlockSize=3,isMinMax=False):
        self.oriData=oriData
        self.dataBlockSize=dataBlockSize
        self.covBlockSize=covBlockSize
        self.binsNumber=binsNumber
        self.minMaxBlockSize=minMaxBlockSize
        self.isMinMax=isMinMax

        self.sizeZ=allDataSize[0]
        self.sizeY=allDataSize[1]
        self.sizeX=allDataSize[2]
        
        self.dim=oriData.shape[0]
        self.incrementalNumber=oriData.shape[1]

        self.singleDistModels:List[singleVariant.singleVariantDist3D]=[]
        self.multiCovModel=None

        self.data=oriData[:,:, 0:self.sizeZ, 0:self.sizeY, 0:self.sizeX]

        #print(self.data.shape)
    
    def fit(self):
        ### fit covariance matrix ###
        self.multiCovModel=multiCov(self.data,self.covBlockSize)
        self.multiCovModel.fit()
        ### end ###

        ### fit singleVariantModel ###
        for vIndex in range(self.dim):
            
            _,binEdges=np.histogram(self.oriData[vIndex,:,:,:,:],bins=self.binsNumber)
            data=self.data[vIndex,:,:,:,:]
            model=singleVariant.singleVariantDist3D(data,self.dataBlockSize,self.binsNumber,binEdges,self.minMaxBlockSize,self.isMinMax)
            self.singleDistModels.append(model)
        
        #for i in range(len(self.singleDistModels)):
        #    self.singleDistModels[i].fit()
        
        with Pool(4) as pool:
            self.singleDistModels=pool.map(self.fit_model,self.singleDistModels)

    @staticmethod
    def fit_model(model:singleVariant.singleVariantDist3D):
        model.fit()
        return model
        ### end ###
    
    def saveModel(self, FileName):
        ## 刪除訓練資料 ##
        self.data=[]
        self.oriData=[]
        for i in range(len(self.singleDistModels)):
            self.singleDistModels[i].clearData()
        folderName="model"
        os.makedirs(folderName,exist_ok=True)
        filePath=os.path.join(folderName,f"{FileName}.pkl")
        with open(filePath,"wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load(FileName) -> "multiDistCopula3D":
        folderName="model"
        os.makedirs(folderName,exist_ok=True)
        filePath=os.path.join(folderName,f"{FileName}.pkl")
        with open(filePath,"rb") as f:
            return pickle.load(f)
        
    def saveInfoToFile(self,FileName):

        folderName="model"
        os.makedirs(folderName,exist_ok=True)

        filePath=os.path.join(folderName,f"{FileName}.txt")
        with open(filePath,"a") as f:

            ## Normal info ##
            print(f"Data Size:{self.sizeX}",file=f)
            print(f"BinNumbers:{self.binsNumber}",file=f)
            print(f"Variable Number:{self.dim}",file=f)
            print(f"Cov Block Size:{self.covBlockSize}",file=f)
            ## detail info ##
            for i in range(len(self.singleDistModels)):
                counts = self.singleDistModels[i].getNonZeroBinsCount()
                blocksCounts=self.singleDistModels[i].getBlocksCounts()
                print(f"v{i} non zero bins count: {counts/blocksCounts}",file=f)
    def getNonZeroBinsCounts(self):

        for i in range(len(self.singleDistModels)):
            counts = self.singleDistModels[i].getNonZeroBinsCount()
            blocksCounts=self.singleDistModels[i].getBlocksCounts()
            print(f"v{i} non zero bins count: {counts/blocksCounts}")
        
    def getPointProb(self,vIndex,z,y,x):
        prob=self.singleDistModels[vIndex].getPointProb(z,y,x)
        return prob
    
    def getPointCovMatrix(self,z,y,x):
        covMatrix=self.multiCovModel.getPointCov(z,y,x)
        return covMatrix
    
    def getPointCorrMatrix(self,z,y,x):
        corrMatrix=self.multiCovModel.getPointCorr(z,y,x)
        return corrMatrix
    
    def sampleByPos(self,z,y,x,size=1000):
        corrMatrix = self.getPointCorrMatrix(z,y,x)
        means=np.zeros(self.dim)

        multiVariateModel=multivariate_normal(means,corrMatrix,allow_singular=True)
        samples=multiVariateModel.rvs(size=size)

        for vIndex in range(self.dim):
            cdfs=norm.cdf(samples[:,vIndex])
            histModel=self.singleDistModels[vIndex].getPointHist(z,y,x)

            result=histModel.ppf(cdfs)
            samples[:,vIndex]=result
        
        return samples
    
    def correlationGT2D(self,sizeZ=15,sizeY=15,sizeX=15):

        corrGT=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)
        p_valueGT=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)

        for z in range(sizeZ):
            for y in range(sizeY):
                for x in range(sizeX):
                    v1=self.oriData[0,:,z,y,x]
                    v2=self.oriData[1,:,z,y,x]

                    corr,p_value=pearsonr(v1,v2)
                    corrGT[z,y,x]=corr
                    p_valueGT[z,y,x]=p_value
        
        corrGT.tofile("experiment/corr_gt.bin")
        p_valueGT.tofile("experiment/pValue_gt.bin")
    
    def correlation2D(self,sizeZ=15,sizeY=15,sizeX=15):
        corrs=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)
        p_values=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)

        for z in range(sizeZ):
            for y in range(sizeY):
                for x in range(sizeX):
                    samples=self.sampleByPos(z,y,x)
                    v1=samples[:,0]
                    v2=samples[:,1]

                    corr,p_value=pearsonr(v1,v2)
                    corrs[z,y,x]=corr
                    p_values[z,y,x]=p_value
        
        corrs.tofile("experiment/corrs.bin")
        p_values.tofile("experiment/pValues.bin")
    
    def calProbByConditionGT(self,conditions:NDArray[np.float32],sizeZ=15,sizeY=15,sizeX=15):
        probsGT=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)

        for z in range(sizeZ):
            for y in range(sizeY):
                for x in range(sizeX):
                    dataList=[]
                    for dim in range(self.dim):
                        dataList.append(np.array(self.oriData[dim,:,z,y,x]))
                    
                    mask=np.ones(len(dataList[0]),dtype=bool)

                    for data,(low,high) in zip(dataList,conditions):
                        mask&=(data>=low)& (data<=high)
                    
                    count=np.sum(mask)
                    prob=count/self.incrementalNumber

                    probsGT[z,y,x]=prob
        
        probsGT.tofile("experiment/probGT.bin")
    
    def calProbByCondition(self,conditions:NDArray[np.float32],sizeZ=15,sizeY=15,sizeX=15,sampleSize=1000):
        probs=np.zeros((sizeZ,sizeY,sizeX),dtype=np.float32)

        for z in range(sizeZ):
            for y in range(sizeY):
                for x in range(sizeX):
                    samples=self.sampleByPos(z,y,x,size=sampleSize)
                    dataList=[]
                    for dim in range(self.dim):
                        dataList.append(np.array(samples[:,dim]))
                    
                    mask=np.ones(len(dataList[0]),dtype=bool)

                    for data,(low,high) in zip(dataList,conditions):
                        mask&=(data>=low)& (data<=high)
                    
                    count=np.sum(mask)
                    prob=count/sampleSize

                    probs[z,y,x]=prob
        
        probs.tofile("experiment/prob.bin")

    def vis2DHistogram(self, posZ,posY,posX):

        result=self.sampleByPos(posZ,posY,posX)
        print(result.shape)

        xedges=self.singleDistModels[0].binEdges
        yedges=self.singleDistModels[1].binEdges

        x=self.oriData[0,:,posZ,posY,posX]
        y=self.oriData[1,:,posZ,posY,posX]

        histGT, edeges_used, yedges_used=np.histogram2d(x,y,bins=[xedges,yedges])

        hist, edeges_used, yedges_used=np.histogram2d(result[:,0],result[:,1],bins=[xedges,yedges])

        # 建立兩個子圖：左右排列 (1 row, 2 columns)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 第一張圖：2D histogram
        im = axes[0].imshow(histGT.T, origin='lower',
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            aspect='auto', cmap='viridis')
        axes[0].set_title(f'2D Histogram Ground Truth \n PosZ: {posZ}, PosY: {posY}, PosX: {posX}, Bin: {self.binsNumber}' )
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        fig.colorbar(im, ax=axes[0], label='Counts')

        # 第二張圖：X 的 marginal histogram
        im2 = axes[1].imshow(hist.T, origin='lower',
                            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                            aspect='auto', cmap='viridis')
        axes[1].set_title('2D Histogram ours')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        fig.colorbar(im2, ax=axes[1], label='Counts')

        plt.tight_layout()
        plt.show()

    
    def vis2DScatterplot(self,posZ,posY,posX,size=1000):

        v1GT=self.oriData[0,:,posZ,posY,posX]
        v2GT=self.oriData[1,:,posZ,posY,posX]

        samples=self.sampleByPos(posZ,posY,posX,size)
        v1=samples[:,0]
        v2=samples[:,1]

        plt.figure(figsize=(6, 5))
        plt.scatter(v1GT, v2GT, c='blue', alpha=0.7, edgecolors='k', label='ground Truth')
        plt.scatter(v1, v2, c='red', alpha=0.3, edgecolors='k', label='ours')

        plt.xlabel("Gravity ")
        plt.ylabel("Density ")
        plt.title(f"Gravity vs Density (Two Datasets) \n PosZ: {posZ}, PosY: {posY}, PosX: {posX}, Bin: {self.binsNumber}")
        plt.legend()
        plt.grid(True)
        plt.show()