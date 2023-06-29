import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np

ndvi_freq = np.zeros((8,1000))
ndvi_freq[0,:] = io.loadmat("forest_ndvi.mat")["forest_ndvi"]
ndvi_freq[1,:] = io.loadmat("shrubland_ndvi.mat")["shrubland_ndvi"]
ndvi_freq[2,:] = io.loadmat("grassland_ndvi.mat")["grassland_ndvi"]
ndvi_freq[3,:] = io.loadmat("wetland_ndvi.mat")["wetland_ndvi"]
ndvi_freq[4,:] = io.loadmat("cropland_ndvi.mat")["cropland_ndvi"]
ndvi_freq[5,:] = io.loadmat("urban_ndvi.mat")["urban_ndvi"]
ndvi_freq[6,:] = io.loadmat("barren_ndvi.mat")["barren_ndvi"]
ndvi_freq[7,:] = io.loadmat("water_ndvi.mat")["water_ndvi"]

ndwi_freq = np.zeros((8,1000))
ndwi_freq[0,:] = io.loadmat("forest_ndwi.mat")["forest_ndwi"]
ndwi_freq[1,:] = io.loadmat("shrubland_ndwi.mat")["shrubland_ndwi"]
ndwi_freq[2,:] = io.loadmat("grassland_ndwi.mat")["grassland_ndwi"]
ndwi_freq[3,:] = io.loadmat("wetland_ndwi.mat")["wetland_ndwi"]
ndwi_freq[4,:] = io.loadmat("cropland_ndwi.mat")["cropland_ndwi"]
ndwi_freq[5,:] = io.loadmat("urban_ndwi.mat")["urban_ndwi"]
ndwi_freq[6,:] = io.loadmat("barren_ndwi.mat")["barren_ndwi"]
ndwi_freq[7,:] = io.loadmat("water_ndwi.mat")["water_ndwi"]

bsi_freq = np.zeros((8,1000))
bsi_freq[0,:] = io.loadmat("forest_bsi.mat")["forest_bsi"]
bsi_freq[1,:] = io.loadmat("shrubland_bsi.mat")["shrubland_bsi"]
bsi_freq[2,:] = io.loadmat("grassland_bsi.mat")["grassland_bsi"]
bsi_freq[3,:] = io.loadmat("wetland_bsi.mat")["wetland_bsi"]
bsi_freq[4,:] = io.loadmat("cropland_bsi.mat")["cropland_bsi"]
bsi_freq[5,:] = io.loadmat("urban_bsi.mat")["urban_bsi"]
bsi_freq[6,:] = io.loadmat("barren_bsi.mat")["barren_bsi"]
bsi_freq[7,:] = io.loadmat("water_bsi.mat")["water_bsi"]

ndmi_freq = np.zeros((8,1000))
ndmi_freq[0,:] = io.loadmat("forest_ndmi.mat")["forest_ndmi"]
ndmi_freq[1,:] = io.loadmat("shrubland_ndmi.mat")["shrubland_ndmi"]
ndmi_freq[2,:] = io.loadmat("grassland_ndmi.mat")["grassland_ndmi"]
ndmi_freq[3,:] = io.loadmat("wetland_ndmi.mat")["wetland_ndmi"]
ndmi_freq[4,:] = io.loadmat("cropland_ndmi.mat")["cropland_ndmi"]
ndmi_freq[5,:] = io.loadmat("urban_ndmi.mat")["urban_ndmi"]
ndmi_freq[6,:] = io.loadmat("barren_ndmi.mat")["barren_ndmi"]
ndmi_freq[7,:] = io.loadmat("water_ndmi.mat")["water_ndmi"]

savi_freq = np.zeros((8,1000))
savi_freq[0,:] = io.loadmat("forest_savi.mat")["forest_savi"]
savi_freq[1,:] = io.loadmat("shrubland_savi.mat")["shrubland_savi"]
savi_freq[2,:] = io.loadmat("grassland_savi.mat")["grassland_savi"]
savi_freq[3,:] = io.loadmat("wetland_savi.mat")["wetland_savi"]
savi_freq[4,:] = io.loadmat("cropland_savi.mat")["cropland_savi"]
savi_freq[5,:] = io.loadmat("urban_savi.mat")["urban_savi"]
savi_freq[6,:] = io.loadmat("barren_savi.mat")["barren_savi"]
savi_freq[7,:] = io.loadmat("water_savi.mat")["water_savi"]

classnames = ["Forest","Shrubland","Grassland","Wetland","Cropland","Urban","Barren","Water"]
colors = ["#009900","#C6B044","#B6FF05","#27FF87","#C24F44","#A5A5A5","#F9FFA4","#1C0DFF"]
x = np.linspace(-1,1,1000)

fontsize_title = 48
fontsize = 32
plt.rcParams['savefig.dpi'] = 1600 
plt.rcParams['savefig.transparent'] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.figure(figsize=(12,6))
for i in range(8):
    plt.plot(x,ndvi_freq[i,:],color=colors[i])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.title("NDVI",fontsize=fontsize_title)
plt.savefig("NDVI.png", facecolor=(1,1,1,0), bbox_inches='tight')

plt.figure(figsize=(12,6))
for i in range(8):
    plt.plot(x,ndwi_freq[i,:],color=colors[i])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.title("NDWI",fontsize=fontsize_title)
plt.savefig("NDWI.png", facecolor=(1,1,1,0), bbox_inches='tight')

plt.figure(figsize=(12,6))
for i in range(8):
    plt.plot(x,bsi_freq[i,:],color=colors[i])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.title("BSI",fontsize=fontsize_title)
plt.savefig("BSI.png", facecolor=(1,1,1,0), bbox_inches='tight')

plt.figure(figsize=(12,6))
for i in range(8):
    plt.plot(x,ndmi_freq[i,:],color=colors[i])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.title("NDMI",fontsize=fontsize_title)
plt.savefig("NDMI.png", facecolor=(1,1,1,0), bbox_inches='tight')

plt.figure(figsize=(12,6))
for i in range(8):
    plt.plot(x,savi_freq[i,:],color=colors[i])
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,
    right=False,
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
plt.title("SAVI",fontsize=fontsize_title)
plt.savefig("SAVI.png", facecolor=(1,1,1,0), bbox_inches='tight')