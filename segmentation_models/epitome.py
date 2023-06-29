import torch as T
import numpy as np
import tifffile as tiff
import models
import scipy.io as io

mu = io.loadmat('mu_training.mat')['mu_training']
samples = 200000
batch_size = 64
lc_class = [1,2,4,5,6,7,9,10]

def lc_pred(r):
  pred = np.zeros(r.shape[2:4])
  for x in range(r.shape[2]):
    for y in range(r.shape[3]):
      pred[x,y] = lc_class[np.argmax(r[:,0,x,y])]
  return pred

T.set_grad_enabled(False)

# compute low-res embedding
def lr_embed(image, lowres, patch_size, max_iter=201, batch_size=64):
    # set up epitome model with uniform variance
    ep = models.EpitomeModel(image.shape[-1],4,1).to(device)
    ep.prior[:] = 0.
    ep.mean[0] = T.from_numpy(image)
    ep.ivar[:] = 100.

    # initialize counters
    lrmap = T.zeros((8,ep.layers,ep.size,ep.size)).to(device)
    
    for it in range(max_iter):
        #construct batches
        batch = np.zeros((batch_size,4,patch_size,patch_size))
        lc_ctr = np.zeros((batch_size,), dtype=int)

        for b in range(batch_size):
            x = np.random.randint(image.shape[1]-patch_size+1)
            y = np.random.randint(image.shape[2]-patch_size+1)
            if int(lowres[x+patch_size//2,y+patch_size//2]) == 8:
                continue
            else:
                batch[b] = image[:,x:x+patch_size,y:y+patch_size]
                lc_ctr[b] = int(lowres[x+patch_size//2,y+patch_size//2])
        x = T.from_numpy(batch).to(device,T.float)

        # compute smoothed posterior and embed
        posterior = (ep(x).view(batch_size,-1) / patch_size**2).softmax(1).view(batch_size,ep.layers,ep.size,ep.size)
        for j in range(batch_size):
            lrmap[lc_ctr[j]] += posterior[j]

    return lrmap.cpu().numpy()
    
# super-resolution EM algorithm
def superres(lrmap, max_iter=20, eps=0.000000001):
    
    # use the full p(l,c)
    p_l_c = T.from_numpy(mu).float().to(device)

    # init the p(s|c): renormalize priors, then normalize over positions
    p_s_c = (T.from_numpy(lrmap).float() + eps).to(device)
    p_s_c /= (p_s_c.sum(0))
    p_s_c /= p_s_c.sum((1,2,3)).view(-1,1,1,1)

    # init the p(l|s) and q(s|l,c)
    p_l_s = (T.rand(p_s_c.shape[1:]+(8,))+10).to(device)
    p_l_s /= p_l_s.sum(3).unsqueeze(3)
    q = T.empty(p_s_c.shape[1:] + p_l_c.shape)

    for it in range(max_iter):
        # E step
        q = T.einsum('exyl,cexy->exycl',p_l_s,p_s_c) + eps
        q /= q.sum((0,1,2))

        # M step
        p_l_s = T.einsum('exycl,cl->exyl',q,p_l_c)+eps
        p_l_s /= p_l_s.sum(3).unsqueeze(3)

    return p_l_s.cpu().numpy()

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

    device = T.device('cuda')
    for n in range(15):
        s2_list = []; lc_list = []; dfc_list = []
        for i in range(n*64,(n+1)*64):
            s2 = tiff.imread("../validation/validation_image/ROIs0000_validation_s2_0_p"+str(i)+".tif").transpose((2,0,1))
            lc = tiff.imread("../validation/validation_LR/ROIs0000_validation_lc_0_p"+str(i)+".tif").transpose((2,0,1))
            s2_list.append(s2); lc_list.append(lc)

        reference_list = list(range(64))

        s2 = con(s2_list,8,8,2,1,reference_list)
        lc = con(lc_list,8,8,2,1,reference_list)
        s2 = (s2[[3,2,1,7],:,:])/10000
        s2[s2 > 1] = 1
        lc = lc[0,:,:]

        lc[lc==1]=0; lc[lc==2]=0;  lc[lc==3]=0; lc[lc==4]=0; lc[lc==5]=0; lc[lc==8]=0; lc[lc==9]=0 # savanna
        lc[lc==6]=1; lc[lc==7]=1; # Shrubland 
        lc[lc==10]=2 # Grassland
        lc[lc==11]=3 # Wetland
        lc[lc==12]=4; lc[lc==14]=4 # Cropland
        lc[lc==13]=5 # Urban/Built-up
        lc[lc==16]=6 # Barren
        lc[lc==17]=7 # Water
        lc[lc==15]=8; lc[lc==18]=8

        lrmap = lr_embed(s2, lc, 7, samples // 64, 64)
        hrp = superres(lrmap,40,1e-9).transpose(3,0,1,2)
        pred = lc_pred(hrp)
        tiff.imsave("./results/Epitome/con_preds/"+str(n)+".tif",pred)
        
