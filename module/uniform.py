import numpy as np

class uniform3d():
    def __init__(self, low:int, high:int):
        self.low=np.array([low,low,low],dtype=np.float32)
        self.high=np.array([high,high,high],dtype=np.float32)
    
    def getVolume(self):
        
        """
        low = np.asarray(self.low, dtype=float)
        high = np.asarray(self.high, dtype=float)
        assert low.shape == (3,) and high.shape == (3,)
        if not np.all(high > low):
            raise ValueError("Each high must be > low.")
        """

        return np.prod(self.high - self.low)
    
    def getPdf(self):
        v=self.getVolume()
        return 1.0/v
        