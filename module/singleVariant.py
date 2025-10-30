import numpy as np
from numpy.typing import NDArray
from typing import List
from numba import njit, float32
from scipy.stats import multivariate_normal, rv_histogram
from module.uniform import uniform3d

class minMaxBlock():
    def __init__(self,oriData:NDArray[np.float32]):
        self.oriData=oriData
        self.max=float32('-inf')
        self.min=float32('inf')

    def fit(self):
        for i in range(self.oriData.shape[0]):
            if self.oriData[i]>self.max:
                self.max=self.oriData[i]
            if self.oriData[i]<self.min:
                self.min=self.oriData[i]
        
        self.oriData=[]
    def getMinMax(self):
        return self.min, self.max
    
    


class singleBlockDist3D():
    def __init__(self,oriData:NDArray[np.float32],blockSize,binsNumber,binEdges):
        self.oriData=oriData
        self.blockSize=blockSize
        self.binsNumber=binsNumber
        self.binEdges=binEdges

        self.incrementalNumber=self.oriData.shape[0]
        self.hist, _ =np.histogram(self.oriData,bins=self.binEdges)
        self.correlations_p=np.zeros((3,6,self.binsNumber),dtype=np.float32)
        self.uniformModel=uniform3d(0,blockSize)
    
    def clearData(self):
        self.oriData=[]
############# Fit Data ######################

    # Position corrIdx
    #   z y x
    # z 
    # y 0
    # x 1 2
    #@staticmethod
    #@njit
    def correlationIter3D(self,corrIdx,x,y,binIdx):
        #sum_x
        self.correlations_p[corrIdx,0,binIdx]+=float32(x)
        #sum_y
        self.correlations_p[corrIdx,1,binIdx]+=float32(y)
        #sum_x2
        self.correlations_p[corrIdx,2,binIdx]+=float32(x**2)
        #sum_y2
        self.correlations_p[corrIdx,3,binIdx]+=float32(y**2)
        #sum_xy
        self.correlations_p[corrIdx,4,binIdx]+=float32(x*y)
        #ensembleNumber
        self.correlations_p[corrIdx,5,binIdx]+=float32(1)
        
    #@staticmethod
    #@njit
    def fitData(self):

        for i in range(self.incrementalNumber):
            for binIdx in range(self.binsNumber):
                if self.hist[binIdx]==0:
                    continue

                for z in range(self.blockSize):
                    for y in range(self.blockSize):
                        for x in range(self.blockSize):
                                
                            if self.binEdges[binIdx]<=self.oriData[i,z,y,x] and self.oriData[i,z,y,x]<self.binEdges[binIdx+1]:
                                self.correlationIter3D(0,z,y,binIdx)
                                self.correlationIter3D(1,z,x,binIdx)
                                self.correlationIter3D(2,y,x,binIdx)
    
    
############# End Fit Data ######################

############# Reconstruct Data #####################
    def getNonZeroBinsCount(self):
        counts=np.count_nonzero(self.hist)
        return counts 
    
    def getStorageSize(self):
        binCounts=self.getNonZeroBinsCount()
        Size=10*binCounts+2*binCounts

        return Size
    
    def getParmByBinNumber(self,binIdx):
        sum_z = self.correlations_p[0,0,binIdx]
        sum_y = self.correlations_p[0,1,binIdx]
        sum_z2 = self.correlations_p[0,2,binIdx]
        sum_y2 = self.correlations_p[0,3,binIdx]
        sum_x = self.correlations_p[1,1,binIdx]
        sum_x2 = self.correlations_p[1,3,binIdx]
        sum_ZY=self.correlations_p[0,4,binIdx]
        sum_ZX=self.correlations_p[1,4,binIdx]
        sum_YX=self.correlations_p[2,4,binIdx]
        
        result=np.array([sum_z,sum_y,sum_x,sum_z2,sum_y2,sum_x2,sum_ZY,sum_ZX,sum_YX])
        return result

    def getVariance(self,corrIdx,binIdx):
        sum_x = self.correlations_p[corrIdx,0,binIdx]
        sum_y = self.correlations_p[corrIdx,1,binIdx]
        sum_xy = self.correlations_p[corrIdx,4,binIdx]
        counts = self.correlations_p[corrIdx,5,binIdx]
        r = (sum_xy-sum_x*sum_y / counts) / (counts-1)
        return r
    
    def getCovMatirx(self,binIdx):
        covMatrix=np.zeros([3,3]).astype(np.float32)

        sum_z = self.correlations_p[0,0,binIdx]
        sum_y = self.correlations_p[0,1,binIdx]
        sum_z2 = self.correlations_p[0,2,binIdx]
        sum_y2 = self.correlations_p[0,3,binIdx]
        sum_x = self.correlations_p[1,1,binIdx]
        sum_x2 = self.correlations_p[1,3,binIdx]

        counts = self.correlations_p[1,5,binIdx]
        
        covMatrix[0,0] = (sum_z2-sum_z**2 / counts) / (counts-1)
        covMatrix[1,1] = (sum_y2-sum_y**2 / counts) / (counts-1)
        covMatrix[2,2] = (sum_x2-sum_x**2 / counts) / (counts-1)
        
        corrZY=self.getVariance(0,binIdx)
        corrZX=self.getVariance(1,binIdx)
        corrYX=self.getVariance(2,binIdx)
        covMatrix[1,0]=corrZY
        covMatrix[0,1]=corrZY
        covMatrix[0,2]=corrZX
        covMatrix[2,0]=corrZX
        covMatrix[2,1]=corrYX
        covMatrix[1,2]=corrYX

        return covMatrix
    def getCountByBinNumber(self,binNumber):
        return self.correlations_p[1,5,binNumber]
    
    def getAllCounts(self):
        result=np.zeros((self.binsNumber))
        for i in range(self.binsNumber):
            result[i]=self.getCountByBinNumber(i)
        
        return result
    def getMean(self,binNumber):
        counts= self.getCountByBinNumber(binNumber)
        if counts==0:
            return [0,0,0]
        # z,y,x (mean)
        return np.array([self.correlations_p[0,0,binNumber]/counts, self.correlations_p[0,1,binNumber]/counts, self.correlations_p[1,1,binNumber]/counts])
    
    def makePsd(self,covMatrix,epsilon=1e-6):
        eigvals, eigvecs = np.linalg.eigh(covMatrix)
        eigvals_clipped = np.clip(eigvals, a_min=epsilon, a_max=None)
        cov_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        return cov_psd
    
    def getPointPdf(self,z,y,x):
        pdfValues=np.zeros([self.binsNumber],dtype=np.float32)
        for binIdx in range(self.binsNumber):
            if self.hist[binIdx]<2:
                continue
            
            mean=self.getMean(binIdx)
            covMatix=self.getCovMatirx(binIdx)
            eigvals=np.linalg.eigvalsh(covMatix)

            if np.any(eigvals<0):
                covMatix=self.makePsd(covMatix)
            
            """
            ## 走uniform流程
            if self.checkCovMatrixForUniform(covMatix,mean):
                pdfValues[binIdx]=self.uniformModel.getPdf()
                #print("go uniform")
            ## 走gmm流程
            else: 
            """
            mv_gaussian=multivariate_normal(mean=mean , cov=covMatix , allow_singular=True)
            pos=np.array([z,y,x])
            pdfValues[binIdx]=mv_gaussian.pdf(pos)
        
        return pdfValues
    
    def getPointProb(self,z,y,x,minMaxBlock:minMaxBlock=None):
        pdfValues=self.getPointPdf(z,y,x)
        
        prob=pdfValues * self.hist
        prob=prob/np.sum(prob)

        ##做 min, max判斷
        if minMaxBlock is not None:
            minVal,maxVal=minMaxBlock.getMinMax()
            for i in range(self.binsNumber):

                if prob[i]==0:
                    continue
                left=self.binEdges[i]
                right=self.binEdges[i+1]
                
                if right<=minVal or left>=maxVal:
                    prob[i]=0
                    #print("hit")
            
            prob=prob/np.sum(prob)
           
        return prob
    def getMeanBinEdges(self):
        meanBinEdges=np.zeros([self.binsNumber])
        for i in range(self.binsNumber):
            meanBinEdges[i]=(self.binEdges[i] + self.binEdges[i+1]) / 2
        return meanBinEdges
    
    def reconstByPoint(self,z,y,x,minMaxBlock=None):
        meanBinEdges=self.getMeanBinEdges()
    
        prob=self.getPointProb(z,y,x,minMaxBlock)
        mean=(prob*meanBinEdges).sum()


        total=np.sum(prob)
        mean=np.sum(meanBinEdges*prob)/total
        variance=np.sum(prob*(meanBinEdges-mean)**2)/total

        return mean, variance
    
    def getPointHist(self,z,y,x,minMaxBlock:minMaxBlock=None):

        prob=self.getPointProb(z,y,x,minMaxBlock)

        """
        ##做 min, max判斷
        if minMaxBlock is not None:
            minVal,maxVal=minMaxBlock.getMinMax()
            for i in range(self.binsNumber):
                left=self.binEdges[i]
                right=self.binEdges[i+1]
                
                if right<=minVal or left>=maxVal:
                    prob[i]=0
            
            prob=prob/np.sum(prob)
        """    

        dist = prob, self.binEdges
        hist=rv_histogram(dist,density=True)

        return hist
    
    def checkCovMatrixForUniform(self,covMatrix,mean):
        eigvals = np.linalg.eigvals(covMatrix)
        theoretical_var = (self.blockSize**2)/12

        lam_min = np.min(eigvals)
        lam_max = np.max(eigvals)

        if np.allclose(eigvals,theoretical_var,rtol=0.2) and np.allclose(lam_max/lam_min, 1, rtol=0.1) and np.allclose(mean,self.blockSize/2,rtol=0.1):
            return True
        
        return False

############# End Reconstruct Data #####################


class singleVariantDist3D():
    def __init__(self,oriData:NDArray[np.float32],blockSize,binsNumber,binEdges,minMaxBlockSize=3,isMinMax=False):
        self.oriData=oriData
        self.blockSize=blockSize
        self.binsNumber=binsNumber
        #_ , self.binEdges =np.histogram(self.oriData,bins=binsNumber)
        self.binEdges=binEdges
        self.minMaxBlockSize=minMaxBlockSize
        self.isMinMax=isMinMax

        self.incrementalNumber=self.oriData.shape[0]
        self.sizeZ=self.oriData.shape[1]
        self.sizeY=self.oriData.shape[2]
        self.sizeX=self.oriData.shape[3]

        self.blocks:List[singleBlockDist3D]=[]
        self.minMaxBlocks:List[minMaxBlock]=[]
    def clearData(self):
        self.oriData=[]
        for i in range(len(self.blocks)):
            self.oriData=[]
            self.blocks[i].clearData()
    def fit(self):

        for z in range(0, self.sizeZ, self.blockSize):
            for y in range(0, self.sizeY, self.blockSize):
                for x in range(0, self.sizeX, self.blockSize):
                    data=self.oriData[: ,z : z+self.blockSize, y: y+self.blockSize, x: x+self.blockSize]
                    block=singleBlockDist3D(data,self.blockSize,self.binsNumber,self.binEdges)
                    self.blocks.append(block)
        """
        for z in range(0, self.sizeZ, self.minMaxBlockSize):
            for y in range(0, self.sizeY, self.minMaxBlockSize):
                for x in range(0, self.sizeX, self.minMaxBlockSize):
                    data= self.oriData[:, z:z+self.minMaxBlockSize, y:y+self.minMaxBlockSize, x:x+self.minMaxBlockSize].flatten()
                    block=minMaxBlock(data)
                    self.minMaxBlocks.append(block)
        
        for i in range(len(self.minMaxBlocks)):
            self.minMaxBlocks[i].fit()
        """
        for i in range(len(self.blocks)):
            self.blocks[i].fitData()
    
    def getStorageSize(self):
        result=0
        for i in range(len(self.blocks)):
            result+=self.blocks[i].getStorageSize()
        
        return result
    
    def reconst(self):
        
        mean_prob=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        var_prob=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        
        for z in range(self.sizeZ):
            for y in range(self.sizeY):
                for x in range(self.sizeX):
                    minMax=None
                    blockIdx=self.getBlockIdx(z,y,x,self.blockSize)
                    localZ, localY, localX= z%self.blockSize, y%self.blockSize, x%self.blockSize
                    if self.isMinMax:
                        minMaxBlockIdx=self.getBlockIdx(z,y,x,self.minMaxBlockSize)
                        minMax=self.minMaxBlocks[minMaxBlockIdx]
                    mean_prob[z,y,x], var_prob[z, y, x]=self.blocks[blockIdx].reconstByPoint(localZ,localY,localX,minMax)
                    

        
        mean_prob.tofile("experiment/mean_test.bin")
        var_prob.tofile("experiment/var_test.bin")
    
    def reconstGT(self):
        mean_gt=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        var_gt=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)

        mean_binEdges=np.zeros([self.binsNumber])

        for i in range(self.binsNumber):
            mean_binEdges[i]=(self.binEdges[i]+self.binEdges[i+1])/2

        for z in range(self.sizeZ):
            for y in range(self.sizeY):
                for x in range(self.sizeX):
                    #theData=self.oriData[:,z,y,x]
                    #groundTruth, _=np.histogram(theData,bins=self.binEdges)
                    #groundTruth=groundTruth/np.sum(groundTruth)
                    gtProb=self.getPointProbGT(z,y,x)
                    mean_gt[z,y,x]=(gtProb*mean_binEdges).sum()

                    total=np.sum(gtProb)
                    mean=np.sum(mean_binEdges*gtProb)/total
                    variance=np.sum(gtProb*(mean_binEdges-mean)**2)/total
                    var_gt[z,y,x]=variance

        mean_gt.tofile("experiment/mean_gt.bin")
        var_gt.tofile("experiment/var_gt.bin")
    
    def getNonZeroBinsCount(self):
        counts=0
        for i in range(len(self.blocks)):
            counts += self.blocks[i].getNonZeroBinsCount()

        return counts
    
    def getBlocksCounts(self):
        counts = len(self.blocks)
        return counts
    
    ### 現在沒再用 ###
    def getPointProb(self,z,y,x):
        blockWidth=self.sizeZ//self.blockSize
        blockZ, blockY, blockX = z//self.blockSize, y//self.blockSize, x//self.blockSize
        blockIdx = blockZ * (blockWidth**2) + blockY * blockWidth + blockX

        localZ, localY, localX= z%self.blockSize, y%self.blockSize, x%self.blockSize
        prob=self.blocks[blockIdx].getPointProb(localZ,localY,localX)
        return prob
    def getBlockIdx(self,z,y,x,blockSize):
        blockWidthZ=self.sizeZ//blockSize
        blockWidthY=self.sizeY//blockSize
        blockWidthX=self.sizeX//blockSize
    
        blockZ, blockY, blockX = z//blockSize, y//blockSize, x//blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX
        return blockIdx
    
    def getPointHist(self,z,y,x):
        """
        blockWidthZ=self.sizeZ//self.blockSize
        blockWidthY=self.sizeY//self.blockSize
        blockWidthX=self.sizeX//self.blockSize
    
        blockZ, blockY, blockX = z//self.blockSize, y//self.blockSize, x//self.blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX
        """
        blockIdx=self.getBlockIdx(z,y,x,self.blockSize)

        localZ, localY, localX= z%self.blockSize, y%self.blockSize, x%self.blockSize
        
        minMax=None
        
        if self.isMinMax:
            minMaxBlockIdx=self.getBlockIdx(z,y,x,self.minMaxBlockSize)
            minMax=self.minMaxBlocks[minMaxBlockIdx]

        hist=self.blocks[blockIdx].getPointHist(localZ,localY,localX,minMax)
        return hist
    
    def getPointProbGT(self,z,y,x):
        theData = self.oriData[:,z,y,x]
        gtHist, _ = np.histogram(theData,bins=self.binEdges)
        gtProb = gtHist/np.sum(gtHist)

        return gtProb
    
        
