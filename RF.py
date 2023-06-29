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
    img = tifffile.imread("./training/training_image/ROIs0000_test_s2_0_p"+str(i)+".tif").transpose((2,0,1))
    lc = tifffile.imread("./training/training_selection/ROIs0000_test_gc_0_p"+str(i)+".tif")
    img = img[[1,2,3,7,8,11],:,:]
    num = len(img[0,:,:][lc!=0])  
    for j in range(6):
        Xtr[index:index+num,j] = img[j,:,:][lc!=0]
        ytr[index:index+num] = lc[lc!=0] 
    index = index + num

Xtr = Xtr[0:index,:]
ytr = ytr[0:index]

Xtr_shrubland = Xtr[ytr==2]
Xtr_grassland = Xtr[ytr==4]
Xtr_wetland   = Xtr[ytr==5]
Xtr_cropland  = Xtr[ytr==6]
Xtr_urban     = Xtr[ytr==7]
Xtr_barren    = Xtr[ytr==9]
print(np.shape(Xtr_shrubland)[0])
print(np.shape(Xtr_grassland)[0])
print(np.shape(Xtr_wetland)[0])
print(np.shape(Xtr_cropland)[0])
print(np.shape(Xtr_urban)[0])
print(np.shape(Xtr_barren)[0])

class_list = [2,4,5,6,7,9]
Xtrain = np.zeros((3000000,6))
Ytrain = np.zeros(3000000)
index = 0
for i in range(len(class_list)):
    Xtr_class = Xtr[ytr==class_list[i]]
    class_size = int(np.shape(Xtr_class)[0])
    if class_size<=0:
        continue
    if 0<class_size<500000:
        Xtrain[index:index+class_size,:] = Xtr_class
        Ytrain[index:index+class_size] = np.ones((1,class_size))*class_list[i]
        index = index + class_size
    if class_size>=500000:
        rand_num_id = np.random.randint(0,class_size,500000)
        Xtrain[index:index+500000,:] = Xtr_class[rand_num_id,:]
        Ytrain[index:index+500000] = np.ones((1,500000))*class_list[i]
        index = index + 500000

Xtrain = Xtrain[0:index,:]
Ytrain = Ytrain[0:index]

Xtr_shrubland = Xtrain[Ytrain==2]
Xtr_grassland = Xtrain[Ytrain==4]
Xtr_wetland   = Xtrain[Ytrain==5]
Xtr_cropland  = Xtrain[Ytrain==6]
Xtr_urban     = Xtrain[Ytrain==7]
Xtr_barren    = Xtrain[Ytrain==9]

print(np.shape(Xtr_shrubland)[0])
print(np.shape(Xtr_grassland)[0])
print(np.shape(Xtr_wetland)[0])
print(np.shape(Xtr_cropland)[0])
print(np.shape(Xtr_urban)[0])
print(np.shape(Xtr_barren)[0])

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
joblib.dump(clf, 'RF.sav')