import joblib
import numpy as np
import tifffile as tiff
from matplotlib import colors
def mycmap():
    color = []
    color.append([0,153,0])        # 0  Forest
    color.append([198,176,68])     # 1  Shrubland
    color.append([182,255,5])      # 2  Grassland
    color.append([39,255,135])     # 3  Wetland
    color.append([194,79,68])      # 4  Cropland
    color.append([165,165,165])    # 5  Urban
    color.append([249,255,164])    # 6  Barren
    color.append([28,13,255])      # 7  Water
    color.append([0,0,0])          # 8  Void
    color = np.array(color)/255
    cmap = colors.ListedColormap(color)
    return cmap

def classnames():
    return ["FOREST", "SHRUBLAND", "GRASSLAND", "WETLAND",
     "CROPLAND", "URBAN", "BARREN", "WATER", "VOID"]

import matplotlib.patches as mpatches           
def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches

model_name = "RF.sav"
clf = joblib.load(model_name) 
label_list = [2,4,5,6,7,9]

for i in range(5128):
    img = tiff.imread("./training/training_image/ROIs0000_test_s2_0_p"+str(i)+".tif")
    plabel = tiff.imread("./training/training_selection/ROIs0000_test_gc_0_p"+str(i)+".tif")
    dfc = tiff.imread("./training/training_HR/ROIs0000_test_dfc_0_p"+str(i)+".tif")

    plabel_correct = plabel.copy()
    plabel_correct[plabel!=dfc] = 0

    vec_img = img[:,:,[1,2,3,7,8,11]].reshape(65536,6)
    vec_lab = np.copy(plabel.reshape(65536))
    vec_dfc = np.copy(dfc.reshape(65536))
    confidence = clf.predict_proba(vec_img)

    vec_img = np.array(vec_img).astype(float)/10000
    for j in range(65536):
        ndwi = (vec_img[j,1] - vec_img[j,3]) / (vec_img[j,1] + vec_img[j,3])
        ndvi = (vec_img[j,3] - vec_img[j,2]) / (vec_img[j,3] + vec_img[j,2])

        if ndwi > 0:
            vec_lab[j] = 10
            continue

        if ndvi > 0.7:
            vec_lab[j] = 1
            continue

        if vec_lab[j]!=0:
            if vec_lab[j]!=1 and vec_lab[j]!=10:
                if confidence[j, label_list.index(vec_lab[j])] < 0.2 and np.max(confidence[j]) > 0.8:
                    vec_lab[j] = label_list[np.argmax(confidence[j])]
                    continue
        else:
            confidences = confidence[j]
            highest_confidence_class = label_list[np.argmax(confidences)]
            highest_confidence = np.max(confidences)
            second_confidence = np.partition(confidences.flatten(), -2)[-2]

            if highest_confidence_class == 4 or highest_confidence_class == 5:
                if highest_confidence-second_confidence < 0.1 or highest_confidence<0.8:
                    highest_confidence_class = label_list[np.argsort(confidences)[-2]]
                    if highest_confidence_class == 4 or highest_confidence_class == 5:
                        if highest_confidence-second_confidence < 0.1 or highest_confidence<0.8:
                            highest_confidence_class = label_list[np.argsort(confidences)[-3]]
            vec_lab[j] = highest_confidence_class

    rlabel = vec_lab.reshape(256,256)
    tiff.imwrite("./training/training_refined/ROIs0000_test_re_0_p"+str(i)+".tif",rlabel)
    rlabel = rlabel.astype(np.uint8)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['savefig.dpi'] = 300 
    lc = rlabel
    lc[lc==1]=0
    lc[lc==2]=1
    lc[lc==4]=2
    lc[lc==5]=3
    lc[lc==6]=4
    lc[lc==7]=5
    lc[lc==9]=6
    lc[lc==10]=7
    cmap = mycmap()
    refine = (lc) / 8
    refine = cmap(refine)[:, :, 0:3]
    lc = plabel_correct
    lc[lc==0]=8
    lc[lc==1]=0
    lc[lc==2]=1
    lc[lc==4]=2
    lc[lc==5]=3
    lc[lc==6]=4
    lc[lc==7]=5
    lc[lc==9]=6
    lc[lc==10]=7
    correct = (lc) / 8
    correct = cmap(correct)[:, :, 0:3]
    dfc[dfc==1]=0
    dfc[dfc==2]=1
    dfc[dfc==4]=2
    dfc[dfc==5]=3
    dfc[dfc==6]=4
    dfc[dfc==7]=5
    dfc[dfc==9]=6
    dfc[dfc==10]=7
    dfc = (dfc) / 8
    dfc = cmap(dfc)[:, :, 0:3]

    image = img 
    image = image[:, :, [3,2,1]]
    image = image.astype(np.float32)
    image_new = image.copy()
    for k in range(3):
        img0 = image[:,:,k]
        img_max = np.percentile(np.reshape(img0 ,-1),98)
        img_min = np.percentile(np.reshape(img0 ,-1),2)
        img0[img0 > img_max] = img_max
        img0[img0 < img_min] = img_min
        img0 = (img0 - img_min)/(img_max - img_min)
        image_new[:,:,k] = img0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
    ax1.imshow(np.clip(image_new, 0, 1)) 
    ax1.set_title("input")
    ax1.axis("off")
    ax2.imshow(correct)
    ax2.set_title("correct")
    ax2.axis("off")
    ax3.imshow(refine)
    ax3.set_title("refine")
    ax3.axis("off")
    ax4.imshow(dfc)
    ax4.set_title("HR")
    ax4.axis("off")

    lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                    handles=mypatches(), ncol=2, title="Land Cover Classes")
    ttl = fig.suptitle('Selected and Refined labels', y=0.75)
    plt.savefig("./training/training_refined_preview/ROIs0000_test_re_0_p"+str(i)+".tif", bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
    plt.close()

    if i % 100 == 0:
        print("Saved image" + str(i))