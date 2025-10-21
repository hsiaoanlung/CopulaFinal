import numpy as np
from numpy.typing import NDArray
from typing import List
import cupy as cp
import gc
class multiHistogramBlock():
    def __init__(self,oriData:NDArray[np.float32],binEdges:NDArray[np.float32],):
        self.oriData=oriData
        self.binEdges=binEdges
        self.dim=oriData.shape[0]
        self.oriData=self.oriData.reshape(self.dim,-1).T
    def fit(self):

        self.hist,_=np.histogramdd(self.oriData,bins=self.binEdges,)

    def clear(self):
        self.hist=[]

    def sample_from_histogramdd(self, n_samples=100, method="uniform"):
        """
        從 np.histogramdd 的結果中抽樣
        
        Parameters
        ----------
        hist : ndarray
            np.histogramdd 的計數結果
        edges : list of arrays
            每個維度的 bin edges
        n_samples : int
            要抽幾個樣本
        method : str
            "center" = 抽 bin 中心
            "uniform" = 在 bin 區間內均勻抽
        
        Returns
        -------
        samples : ndarray, shape (n_samples, n_dim)
            抽樣結果
        """
        # 轉成機率分布
        pdf = self.hist.flatten().astype(float)
        pdf /= pdf.sum()
        

        # 建立所有 bin 的 index 組合
        bin_indices = np.indices(self.hist.shape,dtype=np.int16).reshape(self.hist.ndim, -1).T

        # 按機率抽樣 bin
        chosen_bins = bin_indices[np.random.choice(len(pdf), size=n_samples, p=pdf)]

        # 轉換成實際座標
        samples = []
        for idx in chosen_bins:
            coords = []
            for dim, bin_idx in enumerate(idx):
                left, right = self.binEdges[dim][bin_idx], self.binEdges[dim][bin_idx + 1]
                if method == "center":
                    coords.append((left + right) / 2)
                elif method == "uniform":
                    coords.append(np.random.uniform(left, right))
            samples.append(coords)

        return np.array(samples)
    
    def sample_from_histogramdd_gpu(self, n_samples, safety_factor=0.6):
        """
        GPU 版本的多維直方圖抽樣
        - 自動偵測 GPU 記憶體並調整 batch size
        - 使用 unravel_index 取代 bin_indices 節省記憶體
        """
        # --- 1️⃣ 準備 PDF ---
        hist_gpu = cp.asarray(self.hist, dtype=cp.float32)
        pdf = hist_gpu.ravel()
        pdf_sum = pdf.sum()
        if pdf_sum == 0:
            raise ValueError("Histogram sum is zero, cannot sample.")
        pdf /= pdf_sum

        ndim = hist_gpu.ndim
        edges_gpu = [cp.asarray(e) for e in self.binEdges]

        # --- 2️⃣ 估算 batch size ---
        mem_info = cp.cuda.runtime.memGetInfo()
        free_mem = mem_info[0]  # 可用記憶體 bytes
        # 預估每筆樣本約消耗 ndim * 8 bytes（float64），再乘安全係數
        bytes_per_sample = ndim * 8 * 4  # 高估一點（含中間變數）
        est_batch = int(free_mem * safety_factor / bytes_per_sample)
        batch_size = max(1, min(n_samples, est_batch))

        print(f"[GPU Sampling] Free mem: {free_mem/1e9:.2f} GB, batch_size ≈ {batch_size:,}")

        # --- 3️⃣ 分批抽樣 ---
        all_samples = []
        remaining = n_samples

        while remaining > 0:
            bs = min(batch_size, remaining)
            chosen_flat_idx = cp.random.choice(len(pdf), size=bs, p=pdf)
            multi_indices = cp.array(cp.unravel_index(chosen_flat_idx, hist_gpu.shape)).T  # shape = (bs, ndim)

            lows = cp.stack([edges_gpu[i][multi_indices[:, i]] for i in range(ndim)], axis=1)
            highs = cp.stack([edges_gpu[i][multi_indices[:, i] + 1] for i in range(ndim)], axis=1)
            rand = cp.random.rand(bs, ndim, dtype=cp.float32)
            samples = lows + rand * (highs - lows)
            all_samples.append(samples)

            remaining -= bs
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            print(remaining)

        # --- 4️⃣ 合併結果 ---
        
        return cp.concatenate(all_samples, axis=0)
    
    def countNoneZeroBins(self):

        counts=np.count_nonzero(self.hist)
        return counts 

class multiHistogramModel():
    def __init__(self,oriData:NDArray[np.float32],blockSize,binsNumber):
        self.oriData=oriData
        self.blockSize=blockSize
        self.binsNumber=binsNumber

        self.dim=self.oriData.shape[0]
        self.sizeZ=self.oriData.shape[2]
        self.sizeY=self.oriData.shape[3]
        self.sizeX=self.oriData.shape[4]

        self.blocks:List[multiHistogramBlock]=[]
    
    def fit(self):
        
        ## 所有varaible的binEdges
        
        vBinEdges=[]
        for vIndex in range(self.dim):

            _,binEdges=np.histogram(self.oriData[vIndex,:,:,:,:],bins=self.binsNumber)
            vBinEdges.append(binEdges)

        vBinEdges=np.array(vBinEdges,dtype=np.float32)
        

        for z in range(0, self.sizeZ,self.blockSize):
            for y in range(0, self.sizeY,self.blockSize):
                for x in range(0, self.sizeX, self.blockSize):

                    data=self.oriData[:, :, z: z+self.blockSize, y:y+self.blockSize, x: x+self.blockSize]
                    block=multiHistogramBlock(data,vBinEdges)
                    self.blocks.append(block)
        
        #for i in range(len(self.blocks)):
        #    self.blocks[i].fit()
    
    def getBlockIdx(self,z,y,x,blockSize):
        blockWidthZ=self.sizeZ//blockSize
        blockWidthY=self.sizeY//blockSize
        blockWidthX=self.sizeX//blockSize
    
        blockZ, blockY, blockX = z//blockSize, y//blockSize, x//blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX
        return blockIdx
    
    def sampleByPos(self,z,y,x,size=1000):

        blockIdx=self.getBlockIdx(z,y,x,self.blockSize)
        self.blocks[blockIdx].fit()
        #samples=self.blocks[blockIdx].sample_from_histogramdd(n_samples=size)
        samples=self.blocks[blockIdx].sample_from_histogramdd_gpu(n_samples=size)
        self.blocks[blockIdx].clear()
        return samples
    
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
        
        probs.tofile("experiment/multiHistProb.bin")
    
    def countNoneZeros(self):
        counts=0
        for i in range(len(self.blocks)):
            self.blocks[i].fit()
            counts+=self.blocks[i].countNoneZeroBins()
            self.blocks[i].clear()
            
        counts=counts/len(self.blocks)
        return counts
