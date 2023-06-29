import metrics
import tifffile as tiff
import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches 

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
    color.append([0,0,0])        # 8  Void
    color = np.array(color)/255
    cmap = colors.ListedColormap(color)
    return cmap

def classnames():
    return ["FOREST", "SHRUBLAND", "GRASSLAND", "WETLAND",
     "CROPLAND", "URBAN", "BARREN", "WATER", "VOID"]

def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches

def con(img_list,n,m,a1,a2,rl):
    img = []
    for i in range(n):
        conimg = img_list[rl[i*m]]
        for j in range(1,m):
            conimg = np.concatenate((conimg,img_list[rl[i*m+j]]),axis=a1)
        if i==0:
            img = conimg
        else:
            img = np.concatenate((img,conimg),axis=a2)    
    return img

if __name__ == '__main__':


    conf_mat = metrics.ConfMatrix(8)
    for n in range(15):
        con_pred = tiff.imread("./results/Epitome/con_preds/"+str(n)+".tif")
        for i in range(8):
            for j in range(8):
                pred = con_pred[i*256:(i+1)*256, j*256:(j+1)*256]
                lc = pred
                lc[lc==1]=0
                lc[lc==2]=1
                lc[lc==3]=0 
                lc[lc==4]=2
                lc[lc==5]=3
                lc[lc==6]=4
                lc[lc==7]=5
                lc[lc==9]=6
                lc[lc==10]=7
                HR = tiff.imread("../validation/validation_HR/ROIs0000_validation_dfc_0_p"+str(n*64+i*8+j)+".tif")
                dfc = HR
                dfc[dfc==1]=0
                dfc[dfc==2]=1
                dfc[dfc==3]=0 
                dfc[dfc==4]=2
                dfc[dfc==5]=3
                dfc[dfc==6]=4
                dfc[dfc==7]=5
                dfc[dfc==9]=6
                dfc[dfc==10]=7
                conf_mat.add(dfc, lc)

                output = lc.astype(np.uint8)
                import matplotlib.pyplot as plt
                plt.rcParams['savefig.dpi'] = 300 
                # plt.rcParams["font.family"] = "serif"
                # plt.rcParams["font.serif"] = ["Times New Roman"]
                cmap = mycmap()
                output = (output) / 8
                output = cmap(output)[:, :, 0:3]
                gt = dfc.astype(np.uint8)
                gt = (gt) / 8
                gt = cmap(gt)[:, :, 0:3]
                display_channels = [0,1,2]
                fig, (ax1, ax2) = plt.subplots(1, 2)

                ax1.imshow(output)
                ax1.set_title("prediction")
                ax1.axis("off")
                ax2.imshow(gt)
                ax2.set_title("label")
                ax2.axis("off")
                
                lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                                handles=mypatches(), ncol=2, title="Land Cover Classes")
                plt.savefig( "./results/Epitome/preview/ROIs0000_validation_dfc_0_p"+str(n*64+i*8+j)+".tif", bbox_extra_artists=(lgd,), bbox_inches='tight')
                plt.close()

    state = conf_mat.state
    accuracy = []
    for i in range(8):
        with np.errstate(divide='ignore',invalid='ignore'):
            temp = state[i,i]/state[i].sum()
            temp = np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
            accuracy.append(temp) 

    log_file="./results/Epitome/HR.log"
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

    