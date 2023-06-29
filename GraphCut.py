import maxflow
import numpy as np
import scipy.io as io
import tifffile as tiff
from osgeo import gdal

# Load stats for each index
ndvi_freq = np.zeros((8,1000))
ndvi_freq[0,:] = io.loadmat("./Index_distributions/forest_ndvi.mat")["forest_ndvi"]
ndvi_freq[1,:] = io.loadmat("./Index_distributions/shrubland_ndvi.mat")["shrubland_ndvi"]
ndvi_freq[2,:] = io.loadmat("./Index_distributions/grassland_ndvi.mat")["grassland_ndvi"]
ndvi_freq[3,:] = io.loadmat("./Index_distributions/wetland_ndvi.mat")["wetland_ndvi"]
ndvi_freq[4,:] = io.loadmat("./Index_distributions/cropland_ndvi.mat")["cropland_ndvi"]
ndvi_freq[5,:] = io.loadmat("./Index_distributions/urban_ndvi.mat")["urban_ndvi"]
ndvi_freq[6,:] = io.loadmat("./Index_distributions/barren_ndvi.mat")["barren_ndvi"]
ndvi_freq[7,:] = io.loadmat("./Index_distributions/water_ndvi.mat")["water_ndvi"]

ndwi_freq = np.zeros((8,1000))
ndwi_freq[0,:] = io.loadmat("./Index_distributions/forest_ndwi.mat")["forest_ndwi"]
ndwi_freq[1,:] = io.loadmat("./Index_distributions/shrubland_ndwi.mat")["shrubland_ndwi"]
ndwi_freq[2,:] = io.loadmat("./Index_distributions/grassland_ndwi.mat")["grassland_ndwi"]
ndwi_freq[3,:] = io.loadmat("./Index_distributions/wetland_ndwi.mat")["wetland_ndwi"]
ndwi_freq[4,:] = io.loadmat("./Index_distributions/cropland_ndwi.mat")["cropland_ndwi"]
ndwi_freq[5,:] = io.loadmat("./Index_distributions/urban_ndwi.mat")["urban_ndwi"]
ndwi_freq[6,:] = io.loadmat("./Index_distributions/barren_ndwi.mat")["barren_ndwi"]
ndwi_freq[7,:] = io.loadmat("./Index_distributions/water_ndwi.mat")["water_ndwi"]

bsi_freq = np.zeros((8,1000))
bsi_freq[0,:] = io.loadmat("./Index_distributions/forest_bsi.mat")["forest_bsi"]
bsi_freq[1,:] = io.loadmat("./Index_distributions/shrubland_bsi.mat")["shrubland_bsi"]
bsi_freq[2,:] = io.loadmat("./Index_distributions/grassland_bsi.mat")["grassland_bsi"]
bsi_freq[3,:] = io.loadmat("./Index_distributions/wetland_bsi.mat")["wetland_bsi"]
bsi_freq[4,:] = io.loadmat("./Index_distributions/cropland_bsi.mat")["cropland_bsi"]
bsi_freq[5,:] = io.loadmat("./Index_distributions/urban_bsi.mat")["urban_bsi"]
bsi_freq[6,:] = io.loadmat("./Index_distributions/barren_bsi.mat")["barren_bsi"]
bsi_freq[7,:] = io.loadmat("./Index_distributions/water_bsi.mat")["water_bsi"]

ndmi_freq = np.zeros((8,1000))
ndmi_freq[0,:] = io.loadmat("./Index_distributions/forest_ndmi.mat")["forest_ndmi"]
ndmi_freq[1,:] = io.loadmat("./Index_distributions/shrubland_ndmi.mat")["shrubland_ndmi"]
ndmi_freq[2,:] = io.loadmat("./Index_distributions/grassland_ndmi.mat")["grassland_ndmi"]
ndmi_freq[3,:] = io.loadmat("./Index_distributions/wetland_ndmi.mat")["wetland_ndmi"]
ndmi_freq[4,:] = io.loadmat("./Index_distributions/cropland_ndmi.mat")["cropland_ndmi"]
ndmi_freq[5,:] = io.loadmat("./Index_distributions/urban_ndmi.mat")["urban_ndmi"]
ndmi_freq[6,:] = io.loadmat("./Index_distributions/barren_ndmi.mat")["barren_ndmi"]
ndmi_freq[7,:] = io.loadmat("./Index_distributions/water_ndmi.mat")["water_ndmi"]

savi_freq = np.zeros((8,1000))
savi_freq[0,:] = io.loadmat("./Index_distributions/forest_savi.mat")["forest_savi"]
savi_freq[1,:] = io.loadmat("./Index_distributions/shrubland_savi.mat")["shrubland_savi"]
savi_freq[2,:] = io.loadmat("./Index_distributions/grassland_savi.mat")["grassland_savi"]
savi_freq[3,:] = io.loadmat("./Index_distributions/wetland_savi.mat")["wetland_savi"]
savi_freq[4,:] = io.loadmat("./Index_distributions/cropland_savi.mat")["cropland_savi"]
savi_freq[5,:] = io.loadmat("./Index_distributions/urban_savi.mat")["urban_savi"]
savi_freq[6,:] = io.loadmat("./Index_distributions/barren_savi.mat")["barren_savi"]
savi_freq[7,:] = io.loadmat("./Index_distributions/water_savi.mat")["water_savi"]

# Consistency between a pixel and its LR label
def label_sim(label,ndvi,ndwi,bsi,ndmi,savi):

    prob = 0.0
    ndvi_pos = ((ndvi+1)//0.002).astype(int) 
    ndwi_pos = ((ndwi+1)//0.002).astype(int)
    bsi_pos = ((bsi+1)//0.002).astype(int)
    ndmi_pos = ((ndmi+1)//0.002).astype(int)
    savi_pos = ((savi+1)//0.002).astype(int)

    # Forest
    if label == 1:
        if ndvi_pos>750:
            prob = 1
        else:
            prob = 0
    # Shrubland
    if label == 2:
        if ndvi_pos>600:
            prob = 1
        else:
            prob = 0
    # Grassland
    if label == 4:
        ndwi_freq[2,500:1000] = 0
        ndwi_peek = np.argmax(ndwi_freq[2])
        if ndwi_pos<500 and ndwi_pos>ndwi_peek-50 and ndwi_pos<ndwi_peek+50:
            prob = ndwi_freq[2,ndwi_pos]
        else:
            prob = 0
    # Wetland
    if label == 5:
        if ndvi_pos>625:
            prob = 1
        else:
            prob = 0
    # Croplands
    if label == 6:
        ndvi_freq[4,:500] = 0
        ndvi_peek = np.argmax(ndvi_freq[4])
        if ndvi_pos>500 and ndvi_pos>ndvi_peek-50 and ndvi_pos<ndvi_peek+50:
            prob = ndvi_freq[4,ndvi_pos]
        else:
            prob = 0
    # Urban
    if label == 7:
        bsi_freq[5,:500] = 0
        bsi_peek = np.argmax(bsi_freq[5])
        if ndvi_pos>500 and bsi_pos>bsi_peek-50 and bsi_pos<bsi_peek+50:
            prob = bsi_freq[5,bsi_pos]
        else:
            prob = 0
    # Barren
    if label == 9:
        if bsi_pos>500:
            prob = 1
        else:
            prob = 0
    # Water
    if label == 10:
        if ndwi_pos>500:
            prob = 1
        else:
            prob = 0
    return prob

# Consistency between pixels and LR labels for the whole image
def label_map(img,label):
    img = np.array(img).astype(float)/10000
    lmap = np.zeros((256,256))
    ndvi = (img[7]-img[3])/(img[7]+img[3])
    ndwi = (img[2]-img[7])/(img[2]+img[7])
    ndmi = (img[7]-img[11])/(img[7]+img[11])
    bsi = ((img[3]+img[11])-(img[7]+img[1]))/((img[3]+img[11])+(img[7]+img[1]))
    savi = (img[7]-img[3])/(img[7]+img[3]+0.5)*1.5
    savi[savi>1] = 1
    savi[savi<-1] = -1

    for i in range(256):
        for j in range(256):
            lmap[i,j] = label_sim(label[i,j],ndvi[i,j],ndwi[i,j],bsi[i,j],ndmi[i,j],savi[i,j])
    return lmap

# Similarity between two pixels by RGB
def pixel_sim(pix_a,pix_b):
    theta = np.exp(-np.sqrt(((pix_a[0]-pix_b[0])**2+
                             (pix_a[1]-pix_b[1])**2+
                             (pix_a[2]-pix_b[2])**2)))
    return theta

# Similarity between pixels for each row
def img_row_weight(img):
    weight = np.zeros((256,256))
    for m in range(255):
        for n in range(255):
            weight[m,n] = pixel_sim(img[m,n,:],img[m,n+1,:])
    return weight
    
# Similarity between pixels for each column
def img_col_weight(img):
    weight = np.zeros((256,256))
    for m in range(255):
        for n in range(255):
            weight[m,n] = pixel_sim(img[m,n,:],img[m+1,n,:])
    return weight

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

if __name__ == '__main__':

    for i in range(5128):
        img = gdal.Open("./training/training_image/ROIs0000_test_s2_0_p"+str(i)+".tif").ReadAsArray()
        lc = gdal.Open("./training/training_LR/ROIs0000_test_lc_0_p"+str(i)+".tif").ReadAsArray()
        dfc = gdal.Open("./training/training_HR/ROIs0000_test_dfc_0_p"+str(i)+".tif").ReadAsArray()

        lc = lc[0,]
        lc[lc==2]=1;  lc[lc==3]=1; lc[lc==4]=1; lc[lc==5]=1
        lc[lc==6]=2;  lc[lc==7]=2; lc[lc==8]=1; lc[lc==9]=1
        lc[lc==10]=4; lc[lc==11]=5; lc[lc==12]=6; lc[lc==14]=6
        lc[lc==13]=7; lc[lc==15]=8; lc[lc==16]=9; lc[lc==17]=10

        s2 = np.zeros((256,256,3))
        s2[:,:,0]=img[3,:]; s2[:,:,1]=img[2,:]; s2[:,:,2]=img[1,:]

        # Create the graph.
        g = maxflow.Graph[int]()
        # Add the nodes. nodeids has the identifiers of the nodes in the grid.
        # Note that nodeids.shape == img.shape
        nodeids = g.add_grid_nodes((256,256))
        row_structure = np.array([[0,0,0],[0,0,1],[0,0,0]])
        g.add_grid_edges(nodeids, weights=img_row_weight(s2), structure=row_structure, symmetric=True)
        col_structure = np.array([[0,0,0],[0,0,0],[0,1,0]])
        g.add_grid_edges(nodeids, weights=img_col_weight(s2), structure=col_structure, symmetric=True)
        # Add the terminal edges. The image pixels are the capacities
        # of the edges from the source node. The inverted image pixels
        # are the capacities of the edges to the sink node.
        lmap = label_map(img,lc)*2  # beta = 0.5
        g.add_grid_tedges(nodeids, lmap, 2-lmap)

        # Find the maximum flow.
        g.maxflow()
        # Get the segments of the nodes in the grid.
        sgm = g.get_grid_segments(nodeids)
        # The labels should be 1 where sgm is False and 0 otherwise.
        mask = np.int_(np.logical_not(sgm))
        
        #pseudo label
        plabel = lc*mask 
        tiff.imwrite("./training/training_selection/ROIs0000_test_gc_0_p"+str(i)+".tif",plabel)
        print("Saved image" + str(i))

        plabel = plabel.astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.rcParams['savefig.dpi'] = 300 
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        lc[lc==1]=0
        lc[lc==2]=1
        lc[lc==3]=0
        lc[lc==4]=2
        lc[lc==5]=3
        lc[lc==6]=4
        lc[lc==7]=5
        lc[lc==9]=6
        lc[lc==10]=7
        cmap = mycmap()
        original = (lc) / 8
        original = cmap(original)[:, :, 0:3]
        lc = plabel
        lc[lc==0]=8
        lc[lc==1]=0
        lc[lc==2]=1
        lc[lc==4]=2
        lc[lc==5]=3
        lc[lc==6]=4
        lc[lc==7]=5
        lc[lc==9]=6
        lc[lc==10]=7
        cmap = mycmap()
        select = (lc) / 8
        select = cmap(select)[:, :, 0:3]
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
        image = np.moveaxis(image, 0, -1)
        image = image[:, :, [3,2,1]]
        image = image.astype(np.float32)
        image_new = image.copy()
        for j in range(3):
            img0 = image[:,:,j]
            img_max = np.percentile(np.reshape(img0 ,-1),98)
            img_min = np.percentile(np.reshape(img0 ,-1),2)
            img0[img0 > img_max] = img_max
            img0[img0 < img_min] = img_min
            img0 = (img0 - img_min)/(img_max - img_min)
            image_new[:,:,j] = img0

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
        ax1.imshow(np.clip(image_new, 0, 1)) 
        ax1.set_title("input")
        ax1.axis("off")
        ax2.imshow(original)
        ax2.set_title("original")
        ax2.axis("off")
        ax3.imshow(select)
        ax3.set_title("select")
        ax3.axis("off")
        ax4.imshow(dfc)
        ax4.set_title("HR")
        ax4.axis("off")
        lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0),
                        handles=mypatches(), ncol=2, title="Land Cover Classes")
        ttl = fig.suptitle('Original and Selected labels', y=0.75)
        plt.savefig("./training/training_selection_preview/ROIs0000_test_gc_0_p"+str(i)+".tif", bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')

        plt.close()

