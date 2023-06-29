import os
import random
import numpy as np
import torch
import glob
import tifffile

class dataset(torch.utils.data.Dataset):
    def __init__(self, DATA_FOLDER, args):
        super(dataset, self).__init__()
        
        # List of files
        self.data_files = glob.glob(os.path.join(DATA_FOLDER, '*.tif'))
        self.args = args

    def __len__(self):
        return self.args.num_steps*self.args.batch_size

    @classmethod
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        
        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))
            
        return tuple(results)

    def __getitem__(self, i):

        # Pick a random image
        random_idx = random.randint(0, len(self.data_files) - 1)
        data = np.asarray(tifffile.imread(self.data_files[random_idx]).transpose((2,0,1)), dtype='float32')
        data = data[[3,2,1,7],]
        if self.args.label == 'LR':
            label = np.asarray(tifffile.imread(self.data_files[random_idx].replace('image','LR').replace('s2','lc')).transpose((2,0,1)), dtype='int64')
            lc = label[0,]
            lc[lc==1]=0; lc[lc==2]=0;  lc[lc==3]=0; lc[lc==4]=0; lc[lc==5]=0; lc[lc==8]=0; lc[lc==9]=0 # forest
            lc[lc==6]=1; lc[lc==7]=1; # Shrubland 
            lc[lc==10]=2 # Grassland
            lc[lc==11]=3 # Wetland
            lc[lc==12]=4; lc[lc==14]=4 # Cropland
            lc[lc==13]=5 # Urban/Built-up
            lc[lc==16]=6 # Barren
            lc[lc==17]=7 # Water
            lc[lc==15]=8; lc[lc==18]=8
        else:
            label = np.asarray(tifffile.imread(self.data_files[random_idx].replace('image','refined').replace('s2','re')), dtype='int64')
            lc = label
            lc[lc==1]=0
            lc[lc==2]=1
            lc[lc==4]=2
            lc[lc==5]=3
            lc[lc==6]=4
            lc[lc==7]=5
            lc[lc==9]=6
            lc[lc==10]=7
            
        # Data augmentation
        data_p, label_p = self.data_augmentation(data, lc)

        # Return the torch.Tensor values
        return {'image': torch.from_numpy(data_p), 'label': torch.from_numpy(label_p), "id": os.path.basename(self.data_files[random_idx])}
