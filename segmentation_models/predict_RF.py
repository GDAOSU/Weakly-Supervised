import joblib
import numpy as np
import tifffile as tiff
import metrics
from matplotlib import colors
import os
from tqdm import tqdm

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

LABEL = 'refined'
model_name = "./results/RF" + LABEL+ "/RF.sav"
clf = joblib.load(model_name) 
pbar = tqdm(total=986, desc="[Val]")
conf_mat = metrics.ConfMatrix(8)
for i in range(986):
    img = tiff.imread("../validation/validation_image/ROIs0000_validation_s2_0_p"+str(i)+".tif")
    label = tiff.imread("../validation/validation_HR/ROIs0000_validation_dfc_0_p"+str(i)+".tif")  
    lc = label.astype(np.uint8)
    lc[lc==1]=0
    lc[lc==2]=1
    lc[lc==3]=0
    lc[lc==4]=2
    lc[lc==5]=3
    lc[lc==6]=4
    lc[lc==7]=5
    lc[lc==9]=6
    lc[lc==10]=7
    label = lc
    vec_img = img[:,:,[1,2,3,7,8,11]].reshape(65536,6)
    vec_lab = np.copy(label.reshape(65536))
    confidence = clf.predict_proba(vec_img)
    vec_lab = np.argmax(confidence, 1)
    prediction = vec_lab.reshape(256,256)
    conf_mat.add(label, prediction)
    
    id = "ROIs0000_validation_rf_0_p"+str(i)+".tif"
    import matplotlib.pyplot as plt
    plt.rcParams['savefig.dpi'] = 300 
    cmap = mycmap()
    output = prediction.astype(np.uint8) 
    output = (output) / 8
    output = cmap(output)[:, :, 0:3]
    gt = label.astype(np.uint8)
    gt = (gt) / 8
    gt = cmap(gt)[:, :, 0:3] 
    image = img[:, :, [3,2,1]]
    image = image.astype(np.float32)
    image_new = image.copy()
    for i in range(3):
        img0 = image[:,:,i]
        img_max = np.percentile(np.reshape(img0 ,-1),98)
        img_min = np.percentile(np.reshape(img0 ,-1),2)
        img0[img0 > img_max] = img_max
        img0[img0 < img_min] = img_min
        img0 = (img0 - img_min)/(img_max - img_min)
        image_new[:,:,i] = img0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(np.clip(image_new, 0, 1)) 
    ax1.set_title("input")
    ax1.axis("off")
    ax2.imshow(output)
    ax2.set_title("prediction")
    ax2.axis("off")
    ax3.imshow(gt)
    ax3.set_title("label")
    ax3.axis("off")
    lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                    handles=mypatches(), ncol=2, title="Land Cover Classes")
    ttl = fig.suptitle('Model trained with ' + LABEL + ' labels', y=0.75)
    plt.savefig(os.path.join('./results/RF'+LABEL, 'preview', id), bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
    plt.close()
    pbar.update()

pbar.set_description("[Val] AA: {:.2f}%".format(conf_mat.get_aa() * 100))
pbar.close()
state = conf_mat.state
np.set_printoptions(suppress=True)
accuracy = []
for i in range(8):
    with np.errstate(divide='ignore',invalid='ignore'):
        temp = state[i,i]/state[i].sum()
        temp = np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
        print(temp.round(4))
        accuracy.append(temp) 

log_file= './results/RF' + LABEL + '/HR.log'
message = 'Forest: \t{:.6f}\nShrubland: \t{:.6f}\nGrassland: \t{:.6f}\nWetland: \t{:.6f}\nCropland: \t{:.6f}\nUrban: \t\t{:.6f}\nBarren: \t{:.6f}\nWater: \t\t{:.6f}\nAA: \t\t{:.6f}\n'.format(
        accuracy[0],
        accuracy[1],
        accuracy[2],
        accuracy[3],
        accuracy[4],
        accuracy[5],
        accuracy[6], 
        accuracy[7],
        conf_mat.get_aa())
with open(log_file, "a") as log_file:
    log_file.write('%s\n' % message) 