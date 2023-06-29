import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random
from PIL import Image
import metrics
from loader.datasettest import datasettest
import segmentation_models_pytorch as smp
from hrnet.hrnet import get_seg_model
from hrnet.config import cfg
import fcn8

NUM_CLASSES = 8
GPU = 1
SEED = 2023
MODEL = 'HRNet'
LABEL = 'refined'

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed_all(SEED)

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
    color.append([0,0,0])        # 8  Void
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

def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument(
        "--cfg",
        default="hrnet/HRNet.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    return parser.parse_args()

args = get_arguments()
cfg.merge_from_file(args.cfg)

def main():
    """Create the model and start the training."""

    here = osp.dirname(osp.abspath(__file__))
    out_dir = osp.join(here,'results', MODEL+LABEL)

    if MODEL == 'HRNet':
        cfg.MODEL.EXTRA['INPUT'] = 4
        model = get_seg_model(cfg)

    elif MODEL == 'UNet' :
        model = smp.Unet("resnet18", 
                        classes=args.num_classes, 
                        in_channels=4,
                        encoder_weights=None)

    elif MODEL == 'DeepLabv3+':
        model = smp.DeepLabV3Plus(encoder_name='resnet34',
                    encoder_depth=5,
                    encoder_weights='imagenet',
                    in_channels=4,
                    classes=args.num_classes)

    elif MODEL == 'FCN':
        model = fcn8.FCN8s_library(n_class=NUM_CLASSES)

    device = torch.device('cuda:{}'.format(str(args.gpu)))
    path = osp.join(here,'results', MODEL+LABEL, 'checkpoints', 'best.pth')
    checkpoint = torch.load(path, map_location=str(device))
    model.load_state_dict(checkpoint)
    print('load trained model from ' + path)  
    model.eval()
    model.to(device)

    DATA_FOLDER_val = '../validation/validation_image'
    val_set = datasettest(DATA_FOLDER_val, args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
    val(model, val_loader, args, out_dir, device)

def val(model, dataloader, args, out_dir, device):

    pbar = tqdm(total=len(dataloader), desc="[Val]")
    conf_mat = metrics.ConfMatrix(args.num_classes)

    for i, batch in enumerate(dataloader):
        img, gt = batch['image'], batch['label']
        
        with torch.no_grad():
            pred = model(img.to(device))
            if MODEL == 'HRNet':
                pred = model(img.to(device))
                ph, pw = pred.size(2), pred.size(3)
                h, w = gt.size(1), gt.size(2)
                if ph != h or pw != w:
                    score = F.interpolate(
                            input=pred, size=(h, w), mode='nearest')
                pred = np.argmax(score.cpu().numpy(), axis=1)
            else:
                pred = np.argmax(pred.cpu().numpy(), axis=1)
        conf_mat.add_batch(gt, pred)
        pbar.update()

        id = batch["id"][0]
        output = pred.astype(np.uint8)
        output = np.squeeze(output)
        output_img = Image.fromarray(output)
        output_img.save(os.path.join(out_dir, 'predict', id))

        import matplotlib.pyplot as plt
        plt.rcParams['savefig.dpi'] = 300 
        cmap = mycmap()
        output = (output) / 8
        output = cmap(output)[:, :, 0:3]
        gt = np.squeeze(gt.numpy()).astype(np.uint8)
        gt = (gt) / 8
        gt = cmap(gt)[:, :, 0:3]
        display_channels = [0,1,2]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        image = img.cpu().numpy()    
        image = np.squeeze(image)
        image = np.moveaxis(image, 0, -1)
        image = image[:, :, display_channels]

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
        ttl = fig.suptitle('Model trained with '+ LABEL+ ' labels', y=0.75)
        plt.savefig(os.path.join(out_dir, 'preview', id), bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
        plt.close()

    pbar.set_description("[Val] AA: {:.2f}%".format(conf_mat.get_aa() * 100))
    pbar.close()
    state = conf_mat.state
    accuracy = []
    for i in range(8):
        with np.errstate(divide='ignore',invalid='ignore'):
            temp = state[i,i]/state[i].sum()
            temp = np.nan_to_num(temp, nan=0.0, posinf=0.0, neginf=0.0)
            accuracy.append(temp) 

    log_file=osp.join(out_dir, 'HR.log')
    message = 'Checkpoint: Best\nForest: \t{:.6f}\nShrubland: \t{:.6f}\nGrassland: \t{:.6f}\nWetland: \t{:.6f}\nCropland: \t{:.6f}\nUrban: \t\t{:.6f}\nBarren: \t{:.6f}\nWater: \t\t{:.6f}\nAA: \t\t{:.6f}\n'.format(
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
    return

if __name__ == '__main__':
    main()
