import scipy.io as io
import numpy as np
from osgeo import gdal

count_lc = np.zeros(8)
count_m = np.zeros((8,8))

for i in range(5128):
    lc = gdal.Open("../training/training_LR/ROIs0000_test_lc_0_p"+str(i)+".tif").ReadAsArray()
    dfc = gdal.Open("../training/training_HR/ROIs0000_test_dfc_0_p"+str(i)+".tif").ReadAsArray()
    
    lc = lc[0,:,:]
    lc[lc==1]=0; lc[lc==2]=0;  lc[lc==3]=0; lc[lc==4]=0; lc[lc==5]=0; lc[lc==8]=0; lc[lc==9]=0 # forest
    lc[lc==6]=1; lc[lc==7]=1; # Shrubland 
    lc[lc==10]=2 # Grassland
    lc[lc==11]=3 # Wetland
    lc[lc==12]=4; lc[lc==14]=4 # Cropland
    lc[lc==13]=5 # Urban/Built-up
    lc[lc==16]=6 # Barren
    lc[lc==17]=7 # Water

    dfc[dfc==1]=0; dfc[dfc==3]=0 # forest
    dfc[dfc==2]=1 # Shrubland
    dfc[dfc==4]=2 # Grassland
    dfc[dfc==5]=3 # Wetland
    dfc[dfc==6]=4 # Cropland
    dfc[dfc==7]=5 # Urban/Built-up
    dfc[dfc==9]=6 # Barren
    dfc[dfc==10]=7 # Water

    for j in range(8):
        count_lc[j]+=np.sum(lc==j)

    for m in range(256):
        for n in range(256):
            if lc[m,n]<8:
                count_m[lc[m,n],dfc[m,n]]+=1

mu = np.zeros((8,8))
for i in range(8):
    with np.errstate(divide='ignore',invalid='ignore'):
        mu[i,:] = count_m[i,:]/count_lc[i]        
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

io.savemat('./mu_training.mat',{'mu_training': mu})