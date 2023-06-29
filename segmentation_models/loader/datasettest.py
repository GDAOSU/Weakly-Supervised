import os
import numpy as np
import torch
import glob
import tifffile


class datasettest(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER, args):
        super(datasettest, self).__init__()
        
        # List of files
        self.data_files = glob.glob(os.path.join(DATA_FOLDER, '*.tif'))
        self.args = args

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index):

        data = np.asarray(tifffile.imread(self.data_files[index]).transpose((2,0,1)), dtype='float32')
        label = np.asarray(tifffile.imread(self.data_files[index].replace('image','HR').replace('s2','dfc')), dtype='int64')
        
        lc = label
        lc[lc==1]=0
        lc[lc==2]=1
        lc[lc==3]=0 
        lc[lc==4]=2
        lc[lc==5]=3
        lc[lc==6]=4
        lc[lc==7]=5
        lc[lc==9]=6
        lc[lc==10]=7
        data = data[[3,2,1,7],]

        # Return the torch.Tensor values
        return {'image': torch.from_numpy(data), 'label': torch.from_numpy(lc), "id": os.path.basename(self.data_files[index])}



