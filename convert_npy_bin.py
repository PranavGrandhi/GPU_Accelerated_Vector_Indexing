import numpy as np
import os

npy_path = "./queries_data/query1.npy"
    
data = np.load(npy_path)
    
bin_path = os.path.splitext(npy_path)[0] + ".bin"
    
data.tofile(bin_path)