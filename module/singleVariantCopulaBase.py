import numpy as np
from numpy.typing import NDArray
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import  GaussianKDE, UniformUnivariate
from typing import List
class singleBlockCopula():
    def __init__(self,oriData:NDArray[np.float32],blockSize):
        self.oriData=oriData
        self.blockSize=blockSize

        self.model=GaussianMultivariate()
    def fit(self):
        D, Z, Y, X = self.oriData.shape
        # 建立 index grid（展開為跟 data 一樣大小）
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(Z), np.arange(Y), np.arange(X),
            indexing='ij'
        )

        # 擴展成跟 D 維對齊的 shape
        z_idx = np.broadcast_to(z_idx, (D, Z, Y, X))
        y_idx = np.broadcast_to(y_idx, (D, Z, Y, X))
        x_idx = np.broadcast_to(x_idx, (D, Z, Y, X))

        # flatten 所有的值（變成一堆點）
        flat_value = self.oriData.flatten()
        flat_z = z_idx.flatten()
        flat_y = y_idx.flatten()
        flat_x = x_idx.flatten()

        # Stack 成 shape=(N, 4)，每一列是 [value, z, y, x]
        stacked_data = np.stack([flat_value, flat_z, flat_y, flat_x], axis=1)

        marginals = [
            GaussianKDE(),      # 對 value 做 KDE
            UniformUnivariate(),  # 對 z 做 Uniform
            UniformUnivariate(),  # 對 y 做 Uniform
            UniformUnivariate()   # 對 x 做 Uniform
        ]

        # Step 3: 建立 Gaussian Copula 並指定 marginals
        self.model = GaussianMultivariate(distribution=marginals)

        # Step 4: 拿實際資料去 fit
        self.model.fit(stacked_data)
    
    def reconst(self):

        n_samples = 10000

        # === Sample 數據 ===
        samples = self.model.sample(n_samples)
        samples=samples.to_numpy()
        # shape = (n_samples, 4), columns: [value, z, y, x]

        # voxel 的 index 是 round(x) 如果 x ∈ [0, 4]
        indices = np.round(samples[:, 1:4]).astype(int)  # 只拿 z, y, x

        # 濾掉超出邊界的 sample
        mask = np.all((indices >= 0) & (indices < self.blockSize), axis=1)
        valid_samples = samples[mask]
        valid_indices = indices[mask]
        
        # 建立總和與計數 array
        value_sum = np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)
        value_count = np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.int32)
        value_square_sum = np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)  # 加總 value^2
        # 加總 value
        for val, (z, y, x) in zip(valid_samples[:, 0], valid_indices):
            value_sum[z, y, x] += val
            value_square_sum[z, y, x] += val**2
            value_count[z, y, x] += 1

        # 避免除以 0
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_values = np.divide(value_sum, value_count, where=value_count > 0)
            mean_square = np.divide(value_square_sum, value_count, where=value_count > 0)
            variance_values = mean_square - mean_values**2  # E[x^2] - (E[x])^2

        # === Optional: 組成 [value, z, y, x] 的列表 ===
        result_mean =np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)
        result_var =np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)

        for z in range(self.blockSize):
            for y in range(self.blockSize):
                for x in range(self.blockSize):
                    if value_count[z, y, x] > 0:
                        result_mean[z,y,x]=mean_values[z,y,x]
                        result_var[z,y,x]=variance_values[z,y,x]


        return result_mean, result_var
        #result.tofile("experiment/mean_copulas.bin")
        #result_var.tofile("experiment/var_copulas.bin")

class singleVariantCopula():
    def __init__(self,oriData:NDArray[np.float32],blockSize):
        self.oriData=oriData
        self.blockSize=blockSize
        self.sizeZ=self.oriData.shape[1]
        self.sizeY=self.oriData.shape[2]
        self.sizeX=self.oriData.shape[3]

        self.blocks:List[singleBlockCopula]=[]

    def fit(self):

        for z in range(0, self.sizeZ, self.blockSize):
            for y in range(0, self.sizeY, self.blockSize):
                for x in range(0, self.sizeX, self.blockSize):
                    data=self.oriData[: ,z : z+self.blockSize, y: y+self.blockSize, x: x+self.blockSize]
                    block=singleBlockCopula(data,self.blockSize)
                    self.blocks.append(block)
        
        for i in range(len(self.blocks)):
            self.blocks[i].fit()
    
    def reconst(self):
        blockSize=self.blockSize
        mean_prob=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        var_prob=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        idx=0
        for z in range(0, self.sizeZ, blockSize):
            for y in range(0, self.sizeY, blockSize):
                for x in range(0, self.sizeX, blockSize):
                    mean_prob[z : z+blockSize, y: y+blockSize, x: x+blockSize], var_prob[z : z+blockSize, y: y+blockSize, x: x+blockSize]=self.blocks[idx].reconst()
                    idx+=1

                    
        mean_prob.tofile("experiment/mean_copulas.bin")
        var_prob.tofile("experiment/var_copulas.bin")


class blockMultiVariantCopula():
    def __init__(self,oriData:NDArray[np.float32],blockSize):
        self.oriData=oriData
        self.blockSize=blockSize

        self.model=GaussianMultivariate()
    def fit(self):
        self.V,D, Z, Y, X = self.oriData.shape
        # 建立 index grid（展開為跟 data 一樣大小）
        z_idx, y_idx, x_idx = np.meshgrid(
            np.arange(Z), np.arange(Y), np.arange(X),
            indexing='ij'
        )

        # 擴展成跟 D 維對齊的 shape
        z_idx = np.broadcast_to(z_idx, (D, Z, Y, X))
        y_idx = np.broadcast_to(y_idx, (D, Z, Y, X))
        x_idx = np.broadcast_to(x_idx, (D, Z, Y, X))

        stacked_data=[]
        marginals=[]
        # flatten 所有的值（變成一堆點）
        for v in range(self.V):
            flat_value = self.oriData[v,:].flatten()
            stacked_data.append(flat_value)
            marginals.append(GaussianKDE())

        flat_z = z_idx.flatten()
        stacked_data.append(flat_z)
        marginals.append(UniformUnivariate)# 對 z 做 Uniform

        flat_y = y_idx.flatten()
        stacked_data.append(flat_y)
        marginals.append(UniformUnivariate)# 對 y 做 Uniform

        flat_x = x_idx.flatten()
        stacked_data.append(flat_x)
        marginals.append(UniformUnivariate)# 對 x 做 Uniform

        stacked_data = np.stack(stacked_data, axis=1)

        # Step 3: 建立 Gaussian Copula 並指定 marginals
        self.model = GaussianMultivariate(distribution=marginals)

        # Step 4: 拿實際資料去 fit
        self.model.fit(stacked_data)
    
    def sampleByPos(self,z,y,x,size=1000):

        n_samples=size

        # === Sample 數據 ===
        samples = self.model.sample(n_samples)
        samples=samples.to_numpy()
        # shape = (n_samples, value +z,y,x), columns: [value,value...value, z, y, x]

        # voxel 的 index 是 round(x) 如果 x ∈ [0, 4]
        indices = np.round(samples[:, -3:]).astype(int)  # 只拿 z, y, x
        #print(indices)
        # 濾掉超出邊界的 sample
        mask = np.all((indices >= 0) & (indices < self.blockSize), axis=1)
        valid_samples = samples[mask]
        valid_indices = indices[mask]
        valid_samples=np.array(valid_samples)
        valid_indices=np.array(valid_indices)
        
        resultSamples=[]
        #print(valid_samples.shape)
        #print(valid_indices.shape)
        # 加總 value
        for row in zip(valid_samples[:, 0:self.V], valid_indices):
            data,position=row
            SampleZ,SampleY,SampleX=position

            if SampleZ==z and SampleY==y and SampleX==x:
                resultSamples.append(data)
        
        resultSamples=np.array(resultSamples)

        return resultSamples

    def calProbByCondition(self,conditions:NDArray[np.float32]):

        n_samples = 10000

        # === Sample 數據 ===
        samples = self.model.sample(n_samples)
        samples=samples.to_numpy()
        # shape = (n_samples, value +z,y,x), columns: [value,value...value, z, y, x]

        # voxel 的 index 是 round(x) 如果 x ∈ [0, 4]
        indices = np.round(samples[:, -3:]).astype(int)  # 只拿 z, y, x
        #print(indices)
        # 濾掉超出邊界的 sample
        mask = np.all((indices >= 0) & (indices < self.blockSize), axis=1)
        valid_samples = samples[mask]
        valid_indices = indices[mask]
        valid_samples=np.array(valid_samples)
        valid_indices=np.array(valid_indices)
        # 建立總和與計數 array
        value_sum = np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)
        value_count = np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.int32)
        
        #print(valid_samples.shape)
        #print(valid_indices.shape)
        # 加總 value
        for row in zip(valid_samples[:, 0:self.V], valid_indices):
            data,position=row
            z,y,x=position
            isCount=True
            
            value_count[z,y,x]+=1

            for v in range(self.V):
                low=conditions[v,0]
                high=conditions[v,1]
                if not(low<data[v] and data[v]<high):
                    isCount=False
            
            if isCount:
                value_sum[z,y,x]+=1

        # 避免除以 0
        with np.errstate(divide='ignore', invalid='ignore'):
            prob_values = np.divide(value_sum, value_count, where=value_count > 0)

        # === Optional: 組成 [value, z, y, x] 的列表 ===
        result_prob =np.zeros((self.blockSize, self.blockSize, self.blockSize), dtype=np.float32)

        for z in range(self.blockSize):
            for y in range(self.blockSize):
                for x in range(self.blockSize):
                    if value_count[z, y, x] > 0:
                        result_prob[z,y,x]=prob_values[z,y,x]

        return result_prob

class multiVariantCopulaBase():
    def __init__(self,oriData:NDArray[np.float32],blockSize):
        self.oriData=oriData
        self.blockSize=blockSize
        self.sizeZ=self.oriData.shape[2]
        self.sizeY=self.oriData.shape[3]
        self.sizeX=self.oriData.shape[4]

        self.blocks:List[blockMultiVariantCopula]=[]

    def fit(self):

        for z in range(0, self.sizeZ, self.blockSize):
            for y in range(0, self.sizeY, self.blockSize):
                for x in range(0, self.sizeX, self.blockSize):
                    data=self.oriData[: ,:,z : z+self.blockSize, y: y+self.blockSize, x: x+self.blockSize]
                    block=blockMultiVariantCopula(data,self.blockSize)
                    self.blocks.append(block)
        
        for i in range(len(self.blocks)):
            self.blocks[i].fit()
    
    def calProbByCondition(self,conditions):
        blockSize=self.blockSize
        prob=np.zeros((self.sizeZ, self.sizeY, self.sizeX), dtype=np.float32)
        idx=0
        for z in range(0, self.sizeZ, blockSize):
            for y in range(0, self.sizeY, blockSize):
                for x in range(0, self.sizeX, blockSize):
                    prob[z : z+blockSize, y: y+blockSize, x: x+blockSize]=self.blocks[idx].calProbByCondition(conditions)
                    idx+=1

                    
        prob.tofile("experiment/prob_copulas.bin")
    
    def getBlockIdx(self,z,y,x,blockSize):
        blockWidthZ=self.sizeZ//blockSize
        blockWidthY=self.sizeY//blockSize
        blockWidthX=self.sizeX//blockSize
    
        blockZ, blockY, blockX = z//blockSize, y//blockSize, x//blockSize
        blockIdx = blockZ * (blockWidthY*blockWidthX) + blockY * blockWidthX + blockX
        return blockIdx
    def sampleByPos(self,z,y,x,size=1000):
        blockIdx=self.getBlockIdx(z,y,x,self.blockSize)

        localZ, localY, localX= z%self.blockSize, y%self.blockSize, x%self.blockSize

        samples=self.blocks[blockIdx].sampleByPos(localZ,localY,localX,size)
        return samples