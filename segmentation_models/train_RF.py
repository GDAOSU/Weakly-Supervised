import random
import numpy as np
import tifffile
from sklearn.ensemble import RandomForestClassifier

Xtr = np.zeros((336068608,6))
ytr = np.zeros(336068608)

np.random.seed(2050)
random.seed(2050)

index = 0
for i in range(5128):
    img = np.asarray(tifffile.imread("../training/training_image/ROIs0000_test_s2_0_p"+str(i)+".tif"), dtype='float32')
    lc = np.asarray(tifffile.imread("../training/training_refined/ROIs0000_test_re_0_p"+str(i)+".tif"), dtype='int64')
    # lc = lc[:,:,0]
    # lc[lc==1]=0; lc[lc==2]=0;  lc[lc==3]=0; lc[lc==4]=0; lc[lc==5]=0; lc[lc==8]=0; lc[lc==9]=0 # forest
    # lc[lc==6]=1; lc[lc==7]=1; # Shrubland 
    # lc[lc==10]=2 # Grassland
    # lc[lc==11]=3 # Wetland
    # lc[lc==12]=4; lc[lc==14]=4 # Cropland
    # lc[lc==13]=5 # Urban/Built-up
    # lc[lc==16]=6 # Barren
    # lc[lc==17]=7 # Water

    img = img[:,:,[1,2,3,7,8,11]]
    num = 65536  
    Xtr[index:index+num,:] = img.reshape((num,6))
    ytr[index:index+num] = lc.reshape(-1)
    index = index + num

Xtr = Xtr[0:index,:]
ytr = ytr[0:index]

rand_num_id = np.random.randint(0,index,4000000)
Xtrain = Xtr[rand_num_id,:]
Ytrain = ytr[rand_num_id]

number_of_trees = 500
max_depth = 20
min_samples_leaf = 1000
min_samples_split = 4000

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators = number_of_trees, n_jobs = -1,
                             max_depth = max_depth,
                             min_samples_split = min_samples_split,
                             min_samples_leaf = min_samples_leaf)
                             
clf.fit(Xtrain, Ytrain)
import joblib
joblib.dump(clf, './results/RFrefined/RF.sav')