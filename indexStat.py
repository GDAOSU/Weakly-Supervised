import numpy as np
import scipy.io as io
from osgeo import gdal

forest_ndvi = np.zeros((1000,)); shrubland_ndvi = np.zeros((1000,)); grassland_ndvi = np.zeros((1000,)); wetland_ndvi = np.zeros((1000,))
cropland_ndvi = np.zeros((1000,)); urban_ndvi = np.zeros((1000,)); barren_ndvi = np.zeros((1000,)); water_ndvi = np.zeros((1000,))

forest_ndwi = np.zeros((1000,)); shrubland_ndwi = np.zeros((1000,)); grassland_ndwi = np.zeros((1000,)); wetland_ndwi = np.zeros((1000,)) 
cropland_ndwi = np.zeros((1000,)); urban_ndwi = np.zeros((1000,)); barren_ndwi =np.zeros((1000,)); water_ndwi =np.zeros((1000,))

forest_bsi = np.zeros((1000,)); shrubland_bsi = np.zeros((1000,)); grassland_bsi = np.zeros((1000,)); wetland_bsi = np.zeros((1000,)); 
cropland_bsi = np.zeros((1000,)); urban_bsi = np.zeros((1000,)); barren_bsi = np.zeros((1000,)); water_bsi = np.zeros((1000,)); 

forest_ndmi = np.zeros((1000,)); shrubland_ndmi = np.zeros((1000,)); grassland_ndmi = np.zeros((1000,)); wetland_ndmi = np.zeros((1000,))
cropland_ndmi = np.zeros((1000,)); urban_ndmi = np.zeros((1000,)); barren_ndmi = np.zeros((1000,)); water_ndmi = np.zeros((1000,))

forest_savi = np.zeros((1000,)); shrubland_savi =np.zeros((1000,)); grassland_savi = np.zeros((1000,)); wetland_savi = np.zeros((1000,))
cropland_savi = np.zeros((1000,)); urban_savi = np.zeros((1000,)); barren_savi = np.zeros((1000,)); water_savi = np.zeros((1000,))

for i in range(5128):

    img = gdal.Open("../training/training_image/ROIs0000_test_s2_0_p"+str(i)+".tif").ReadAsArray()
    lc = gdal.Open("../training/training_LR/ROIs0000_test_lc_0_p"+str(i)+".tif").ReadAsArray()
    img = np.array(img).astype(float)/10000
    lc = lc[0,]  
    lc[lc==2]=1;  lc[lc==3]=1; lc[lc==4]=1; lc[lc==5]=1
    lc[lc==6]=2;  lc[lc==7]=2; lc[lc==8]=1; lc[lc==9]=1
    lc[lc==10]=4; lc[lc==11]=5; lc[lc==12]=6; lc[lc==14]=6
    lc[lc==13]=7; lc[lc==15]=8; lc[lc==16]=9; lc[lc==17]=10
    
    ndvi = (img[7]-img[3])/(img[7]+img[3])
    ndwi = (img[2]-img[7])/(img[2]+img[7])
    ndmi = (img[7]-img[11])/(img[7]+img[11])
    bsi = ((img[3]+img[11])-(img[7]+img[1]))/((img[3]+img[11])+(img[7]+img[1]))
    savi = (img[7]-img[3])/(img[7]+img[3]+0.5)*1.5

    ndvi = ((ndvi+1)//0.002).astype(int)
    ndwi = ((ndwi+1)//0.002).astype(int)
    ndmi = ((ndmi+1)//0.002).astype(int)
    bsi = ((bsi+1)//0.002).astype(int)
    savi = ((savi+1)//0.002).astype(int)

    for m in range(256):
        for n in range(256):
            if lc[m,n]==1:
                forest_ndvi[ndvi[m,n]] += 1
                forest_ndwi[ndwi[m,n]] += 1
                forest_bsi[bsi[m,n]] += 1
                forest_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    forest_savi[savi[m,n]] += 1
            if lc[m,n]==2:
                shrubland_ndvi[ndvi[m,n]] += 1
                shrubland_ndwi[ndwi[m,n]] += 1
                shrubland_bsi[bsi[m,n]] += 1
                shrubland_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    shrubland_savi[savi[m,n]] += 1
            if lc[m,n]==4:
                grassland_ndvi[ndvi[m,n]] += 1
                grassland_ndwi[ndwi[m,n]] += 1
                grassland_bsi[bsi[m,n]] += 1
                grassland_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    grassland_savi[savi[m,n]] += 1
            if lc[m,n]==5:
                wetland_ndvi[ndvi[m,n]] += 1
                wetland_ndwi[ndwi[m,n]] += 1
                wetland_bsi[bsi[m,n]] += 1
                wetland_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    wetland_savi[savi[m,n]] += 1
            if lc[m,n]==6:
                cropland_ndvi[ndvi[m,n]] += 1
                cropland_ndwi[ndwi[m,n]] += 1
                cropland_bsi[bsi[m,n]] += 1
                cropland_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    cropland_savi[savi[m,n]] += 1
            if lc[m,n]==7:
                urban_ndvi[ndvi[m,n]] += 1
                urban_ndwi[ndwi[m,n]] += 1
                urban_bsi[bsi[m,n]] += 1
                urban_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    urban_savi[savi[m,n]] += 1
            if lc[m,n]==9:
                barren_ndvi[ndvi[m,n]] += 1
                barren_ndwi[ndwi[m,n]] += 1
                barren_bsi[bsi[m,n]] += 1
                barren_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    barren_savi[savi[m,n]] += 1
            if lc[m,n]==10:
                water_ndvi[ndvi[m,n]] += 1
                water_ndwi[ndwi[m,n]] += 1
                water_bsi[bsi[m,n]] += 1
                water_ndmi[ndmi[m,n]] += 1
                if -1<savi[m,n]<1000: 
                    water_savi[savi[m,n]] += 1
                    
    if i % 100 == 0:
        print("finished img " + str(i) + ".")

#NDVI
forest_ndvi_freq = np.array(forest_ndvi/max(forest_ndvi))
shrubland_ndvi_freq = np.array(shrubland_ndvi/max(shrubland_ndvi))
grassland_ndvi_freq = np.array(grassland_ndvi/max(grassland_ndvi))
wetland_ndvi_freq = np.array(wetland_ndvi/max(wetland_ndvi))
cropland_ndvi_freq = np.array(cropland_ndvi/max(cropland_ndvi))
urban_ndvi_freq = np.array(urban_ndvi/max(urban_ndvi))
barren_ndvi_freq = np.array(barren_ndvi/max(barren_ndvi))
water_ndvi_freq = np.array(water_ndvi/max(water_ndvi))

io.savemat("forest_ndvi.mat",{"forest_ndvi":forest_ndvi_freq})
io.savemat("shrubland_ndvi.mat",{"shrubland_ndvi":shrubland_ndvi_freq})
io.savemat("grassland_ndvi.mat",{"grassland_ndvi":grassland_ndvi_freq})
io.savemat("wetland_ndvi.mat",{"wetland_ndvi":wetland_ndvi_freq})
io.savemat("cropland_ndvi.mat",{"cropland_ndvi":cropland_ndvi_freq})
io.savemat("urban_ndvi.mat",{"urban_ndvi":urban_ndvi_freq})
io.savemat("barren_ndvi.mat",{"barren_ndvi":barren_ndvi_freq})
io.savemat("water_ndvi.mat",{"water_ndvi":water_ndvi_freq})

#NDWI
forest_ndwi_freq = np.array(forest_ndwi/max(forest_ndwi))
shrubland_ndwi_freq = np.array(shrubland_ndwi/max(shrubland_ndwi))
grassland_ndwi_freq = np.array(grassland_ndwi/max(grassland_ndwi))
wetland_ndwi_freq = np.array(wetland_ndwi/max(wetland_ndwi))
cropland_ndwi_freq = np.array(cropland_ndwi/max(cropland_ndwi))
urban_ndwi_freq = np.array(urban_ndwi/max(urban_ndwi))
barren_ndwi_freq = np.array(barren_ndwi/max(barren_ndwi))
water_ndwi_freq = np.array(water_ndwi/max(water_ndwi))

io.savemat("forest_ndwi.mat",{"forest_ndwi":forest_ndwi_freq})
io.savemat("shrubland_ndwi.mat",{"shrubland_ndwi":shrubland_ndwi_freq})
io.savemat("grassland_ndwi.mat",{"grassland_ndwi":grassland_ndwi_freq})
io.savemat("wetland_ndwi.mat",{"wetland_ndwi":wetland_ndwi_freq})
io.savemat("cropland_ndwi.mat",{"cropland_ndwi":cropland_ndwi_freq})
io.savemat("urban_ndwi.mat",{"urban_ndwi":urban_ndwi_freq})
io.savemat("barren_ndwi.mat",{"barren_ndwi":barren_ndwi_freq})
io.savemat("water_ndwi.mat",{"water_ndwi":water_ndwi_freq})

#SAVI
forest_savi_freq = np.array(forest_savi/max(forest_savi))
shrubland_savi_freq = np.array(shrubland_savi/max(shrubland_savi))
grassland_savi_freq = np.array(grassland_savi/max(grassland_savi))
wetland_savi_freq = np.array(wetland_savi/max(wetland_savi))
cropland_savi_freq = np.array(cropland_savi/max(cropland_savi))
urban_savi_freq = np.array(urban_savi/max(urban_savi))
barren_savi_freq = np.array(barren_savi/max(barren_savi))
water_savi_freq = np.array(water_savi/max(water_savi))

io.savemat("forest_savi.mat",{"forest_savi":forest_savi_freq})
io.savemat("shrubland_savi.mat",{"shrubland_savi":shrubland_savi_freq})
io.savemat("grassland_savi.mat",{"grassland_savi":grassland_savi_freq})
io.savemat("wetland_savi.mat",{"wetland_savi":wetland_savi_freq})
io.savemat("cropland_savi.mat",{"cropland_savi":cropland_savi_freq})
io.savemat("urban_savi.mat",{"urban_savi":urban_savi_freq})
io.savemat("barren_savi.mat",{"barren_savi":barren_savi_freq})
io.savemat("water_savi.mat",{"water_savi":water_savi_freq})

#NDMI
forest_ndmi_freq = np.array(forest_ndmi/max(forest_ndmi))
shrubland_ndmi_freq = np.array(shrubland_ndmi/max(shrubland_ndmi))
grassland_ndmi_freq = np.array(grassland_ndmi/max(grassland_ndmi))
wetland_ndmi_freq = np.array(wetland_ndmi/max(wetland_ndmi))
cropland_ndmi_freq = np.array(cropland_ndmi/max(cropland_ndmi))
urban_ndmi_freq = np.array(urban_ndmi/max(urban_ndmi))
barren_ndmi_freq = np.array(barren_ndmi/max(barren_ndmi))
water_ndmi_freq = np.array(water_ndmi/max(water_ndmi))

io.savemat("forest_ndmi.mat",{"forest_ndmi":forest_ndmi_freq})
io.savemat("shrubland_ndmi.mat",{"shrubland_ndmi":shrubland_ndmi_freq})
io.savemat("grassland_ndmi.mat",{"grassland_ndmi":grassland_ndmi_freq})
io.savemat("wetland_ndmi.mat",{"wetland_ndmi":wetland_ndmi_freq})
io.savemat("cropland_ndmi.mat",{"cropland_ndmi":cropland_ndmi_freq})
io.savemat("urban_ndmi.mat",{"urban_ndmi":urban_ndmi_freq})
io.savemat("barren_ndmi.mat",{"barren_ndmi":barren_ndmi_freq})
io.savemat("water_ndmi.mat",{"water_ndmi":water_ndmi_freq})

#BSI
forest_bsi_freq = np.array(forest_bsi/max(forest_bsi))
shrubland_bsi_freq = np.array(shrubland_bsi/max(shrubland_bsi))
grassland_bsi_freq = np.array(grassland_bsi/max(grassland_bsi))
wetland_bsi_freq = np.array(wetland_bsi/max(wetland_bsi))
cropland_bsi_freq = np.array(cropland_bsi/max(cropland_bsi))
urban_bsi_freq = np.array(urban_bsi/max(urban_bsi))
barren_bsi_freq = np.array(barren_bsi/max(barren_bsi))
water_bsi_freq = np.array(water_bsi/max(water_bsi))

io.savemat("forest_bsi.mat",{"forest_bsi":forest_bsi_freq})
io.savemat("shrubland_bsi.mat",{"shrubland_bsi":shrubland_bsi_freq})
io.savemat("grassland_bsi.mat",{"grassland_bsi":grassland_bsi_freq})
io.savemat("wetland_bsi.mat",{"wetland_bsi":wetland_bsi_freq})
io.savemat("cropland_bsi.mat",{"cropland_bsi":cropland_bsi_freq})
io.savemat("urban_bsi.mat",{"urban_bsi":urban_bsi_freq})
io.savemat("barren_bsi.mat",{"barren_bsi":barren_bsi_freq})
io.savemat("water_bsi.mat",{"water_bsi":water_bsi_freq})