import numpy as np
from collections import defaultdict
from numpy.typing import NDArray
from typing import List

class SparseMultiHistogramBlock:
    def __init__(self, bin_edges):
        """
        bin_edges: list of 1D numpy arrays for each dimension (same as np.histogramdd)
        """
        self.bin_edges = [np.asarray(e) for e in bin_edges]
        self.ndim = len(bin_edges)
        self.hist = defaultdict(float)
        self.total_count = 0.0
        self._cdf_cache = None  # for sampling

    """
    def _digitize(self, samples):
        #Convert continuous samples to bin indices, matching np.histogramdd behavior.
        indices = [
            np.searchsorted(self.bin_edges[d], samples[:, d], side='right') - 1
            for d in range(self.ndim)
        ]
        # Clip to ensure valid bin indices
        for d in range(self.ndim):
            indices[d] = np.clip(indices[d], 0, len(self.bin_edges[d]) - 2)
        return np.stack(indices, axis=1)
    """
    def _digitize(self, samples):
        """Convert continuous samples to bin indices."""
        indices = [
            np.clip(np.digitize(samples[:, d], self.bin_edges[d]) - 1, 0, len(self.bin_edges[d]) - 2)
            for d in range(self.ndim)
        ]
        return np.stack(indices, axis=1)
    def add_samples(self, samples, weight=1.0):
        """Accumulate histogram counts for samples (N, D)."""
        indices = self._digitize(samples)
        for idx in map(tuple, indices):
            self.hist[idx] += weight
        self.total_count += len(samples) * weight
        self._cdf_cache = None  # invalidate cache

    def normalize(self):
        """Normalize histogram to form a probability distribution."""
        if self.total_count > 0:
            for k in self.hist:
                self.hist[k] /= self.total_count
            self.total_count = 1.0
        self._cdf_cache = None

    def sample(self, n_samples):
        """Draw samples according to histogram probability."""
        if self._cdf_cache is None:
            keys = np.array(list(self.hist.keys()))
            probs = np.array(list(self.hist.values()), dtype=float)
            probs /= probs.sum()
            self._cdf_cache = (keys, probs)

        keys, probs = self._cdf_cache
        chosen_idx = np.random.choice(len(probs), size=n_samples, p=probs)
        chosen_bins = keys[chosen_idx]

        # Uniform random inside each bin
        lows = np.array([self.bin_edges[d][chosen_bins[:, d]] for d in range(self.ndim)]).T
        highs = np.array([self.bin_edges[d][chosen_bins[:, d] + 1] for d in range(self.ndim)]).T
        r = np.random.rand(n_samples, self.ndim)
        return lows + r * (highs - lows)
    
    def getStorageSize(self):
        nonZeroBin=len(self.hist)
        size=nonZeroBin*(self.ndim+1)

        return size
    
class multiHistogramSpaseModel():
    def __init__(self,oriData:NDArray[np.float32],blockSize,binsNumber):
        self.oriData=oriData
        self.blockSize=blockSize
        self.binsNumber=binsNumber

        self.dim=self.oriData.shape[0]
        self.sizeZ=self.oriData.shape[2]
        self.sizeY=self.oriData.shape[3]
        self.sizeX=self.oriData.shape[4]
        self.vBinEdges=[]
        self.blocks:List[SparseMultiHistogramBlock]=[]
    
    def fit(self):
        
        ## 所有varaible的binEdges
        
        for vIndex in range(self.dim):

            _,binEdges=np.histogram(self.oriData[vIndex,:,:,:,:],bins=self.binsNumber)
            self.vBinEdges.append(binEdges)

        self.vBinEdges=np.array(self.vBinEdges,dtype=np.float32)
        

        for z in range(0, self.sizeZ,self.blockSize):
            for y in range(0, self.sizeY,self.blockSize):
                for x in range(0, self.sizeX, self.blockSize):

                    data=self.oriData[:, :, z: z+self.blockSize, y:y+self.blockSize, x: x+self.blockSize]
                    data=data.reshape(self.dim,-1).T
                    block=SparseMultiHistogramBlock(self.vBinEdges)
                    block.add_samples(data)
                    block.normalize()
                    self.blocks.append(block)
        
    def getBlockIdx(self,z,y,x,blockSize):
        blockWidthZ=self.sizeZ//blockSize
        blockWidthY=self.sizeY//blockSize
        blockWidthX=self.sizeX//blockSize
    
        blockZ, blockY, blockX = z//blockSize, y//blockSize, x//blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX
        return blockIdx
    def getHistByPos(self,z,y,x):
        blockIdx=self.getBlockIdx(z,y,x,self.blockSize)
        hist=self.blocks[blockIdx]
        return hist
    
    def sampleByPos(self,z,y,x,size=1000):

        blockIdx=self.getBlockIdx(z,y,x,self.blockSize)
        samples=self.blocks[blockIdx].sample(n_samples=size)
        return samples
    
    def getStorageSize(self):
        result=0
        for i in range(len(self.blocks)):
            result+=self.blocks[i].getStorageSize()    
        print(f"sparse sum: {result}")

        return result

def rmseForSparseHistogram(hist1, hist2):
    """
    Compute RMSE between two SparseMultiHistogram objects.
    Both must have same bin_edges and dimension.
    """
    assert hist1.ndim == hist2.ndim, "Dim mismatch"
    for e1, e2 in zip(hist1.bin_edges, hist2.bin_edges):
        if not np.allclose(e1, e2):
            raise ValueError("Bin edges mismatch between histograms")

    # 所有出現過的 bin（聯集）
    all_keys = set(hist1.hist.keys()) | set(hist2.hist.keys())

    # 計算平方差
    sq_err = 0.0
    for key in all_keys:
        v1 = hist1.hist.get(key, 0.0)
        v2 = hist2.hist.get(key, 0.0)
        diff = v1 - v2
        sq_err += diff * diff

    # RMSE = sqrt(mean(square error))
    rmse_value = np.sqrt(sq_err / len(all_keys))
    return rmse_value