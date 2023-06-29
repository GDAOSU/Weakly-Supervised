from sklearn import metrics
import tifffile as tiff
import numpy as np

dfctr = np.zeros(336068608)
lctr = np.zeros(336068608)
con_matrix = np.zeros((8,8))
index = 0
 
for i in range(5128):
    val = tiff.imread("./training/training_HR/ROIs0000_test_dfc_0_p"+str(i)+".tif")
    pred = tiff.imread("./training/training_refined/ROIs0000_test_re_0_p"+str(i)+".tif")

    # lc = tiff.imread("./training/training_LR/ROIs0000_test_lc_0_p"+str(i)+".tif")
    # lc = lc[:,:,0]
    # lc[lc==2]=1;  lc[lc==3]=1; lc[lc==4]=1; lc[lc==5]=1
    # lc[lc==6]=2;  lc[lc==7]=2; lc[lc==8]=1; lc[lc==9]=1
    # lc[lc==10]=4; lc[lc==11]=5; lc[lc==12]=6; lc[lc==14]=6
    # lc[lc==13]=7; lc[lc==15]=8; lc[lc==16]=9; lc[lc==17]=10

    dfc_remain = val.copy()
    dfc_remain[pred==0] = 0

    lc_correct = pred.copy()
    lc_correct[pred!=val] = 0

    # lc_correct_removed = lc.copy()
    # lc_correct_removed[pred!=0] = 0
    # lc_correct_removed[lc_correct_removed!=val] = 0

    num = 65536
    dfctr[index:index+num] = dfc_remain.reshape(-1) 
    lctr[index:index+num] = lc_correct.reshape(-1) 

    # con_matrix += metrics.confusion_matrix(val.reshape(65536,1),pred.reshape(65536,1),labels=[1,2,4,5,6,7,9,10])

    # if not np.any(pred):
    #     print("Skiped "+str(i)+".tif")
    #     continue
    # con_matrix += metrics.confusion_matrix(pred.reshape(65536,1),val.reshape(65536,1),labels=[1,2,4,5,6,7,9,10])

    index = index+num
    print("Finished "+str(i)+".tif")

for i in [1,2,4,5,6,7,9,10]:
    print("LC class " + str(i) + " = " + str(len(lctr[lctr==i])))

for i in [1,2,4,5,6,7,9,10]:
    print("DFC class " + str(i) + " = " + str(len(dfctr[dfctr==i])))    

# np.set_printoptions(suppress=True)
# AA = 0
# OA = 0
# for i in range(8):
#     acc = con_matrix[i]/sum(con_matrix[i])
#     AA += con_matrix[i,i]/sum(con_matrix[i])
#     OA += con_matrix[i,i]
#     print(acc.round(4))

# AA = AA/8
# OA = OA/sum(sum(con_matrix))

# print()
# print("AA = "+str(AA.round(4)))
# print("OA = "+str(OA.round(4)))