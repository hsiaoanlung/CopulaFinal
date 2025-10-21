from importlib import reload ,import_module
import module.utilize as utilize
import module.multiHistogramBase as multiHistogramBase
import numpy as np
from numba import njit,jit, float32
import matplotlib.pyplot as plt
reload(utilize)
reload(multiHistogramBase)

attribute_names=np.array(["phi_grav","particle_mass_density","zmom","ymom"])
incremental_number=300
all_ensamble_data=np.empty([0,incremental_number,64,64,64])

for name in attribute_names:
    data=utilize.readFiles(name,incremental_number)
    data=data.reshape(1,incremental_number,64,64,64)
    all_ensamble_data=np.append(all_ensamble_data,data,axis=0)



dataBlockSize=2
binsNumber=128
sizeZ=60
sizeY=60
sizeX=60
oriData=all_ensamble_data
multiHistModel=multiHistogramBase.multiHistogramModel(oriData,blockSize=dataBlockSize,binsNumber=binsNumber)
multiHistModel.fit()

conditions=np.array([[0,1e5],[3e10,5e10]])

samples=multiHistModel.sampleByPos(10,10,10,size=100)

print(samples)