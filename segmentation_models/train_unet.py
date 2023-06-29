import os
import os.path as osp
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import random

import metrics
from loader.dataset import dataset
from loader.datasettest import datasettest
import segmentation_models_pytorch as smp



BATCH_SIZE = 5
NUM_CLASSES = 8
SAVE_PRED_EVERY = 5000
NUM_STEPS = 100001
RESUME = 0
GPU = 0
LABEL = 'refined'

LEARNING_RATE = 0.001
MOMENTUM = 0.9
POWER = 0.9
WEIGHT_DECAY = 0.0001
SEED = 2023

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.cuda.manual_seed_all(SEED)


def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--resume", type=bool, default=RESUME,
                        help="restart the training or not")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    parser.add_argument("--label", type=str, default=LABEL,
                        help="choose training label")
    return parser.parse_args()


args = get_arguments()


def accuracy(pred,gt):
    return 100*float(np.count_nonzero(pred==gt)/gt.size)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    train_params = group_weight(nets)
    optimizer = torch.optim.SGD(
        train_params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    return optimizer


def main():
    """Create the model and start the training."""

    # save checkpoint direction
    here = osp.dirname(osp.abspath(__file__))
    out_dir = osp.join(here,'results', 'UNet'+LABEL)
    if not osp.isdir(out_dir):
        os.makedirs(out_dir) 
    checkpoint_dir=osp.join(out_dir,'checkpoints')
    if not osp.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir) 

    # dataloader
    DATA_FOLDER = '../training/training_image'
    train_set = dataset(DATA_FOLDER, args)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size)
    trainloader_iter = iter(train_loader)

    DATA_FOLDER_val = '../validation/validation_image'
    val_set = datasettest(DATA_FOLDER_val, args)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=4)

    model = smp.Unet("resnet18", 
                    classes=args.num_classes, 
                    in_channels=4,
                    encoder_weights=None)

    device = torch.device('cuda:{}'.format(str(args.gpu)))
    path = osp.join(here,'results', 'UNet'+LABEL, 'checkpoints', 'iter' + str(args.resume) +'.pth')
    i_iter=0
    if args.resume: 
        i_iter=args.resume+1
        checkpoint = torch.load(path, map_location=str(device))
        model.load_state_dict(checkpoint)
        print('load trained model from ' + path)  
    model.train()
    model = model.to(device)

    # loss fuction and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.num_classes, reduction='mean')
    optimizer = create_optimizers(model, args)

    max_iter=args.num_steps
    train_loss=[]
    train_acc=[]
    while i_iter < max_iter:

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        batch = next(trainloader_iter)
        
        im_s, label_s = batch["image"], batch["label"] # 5*4*w*h
        im_s, label_s =im_s.to(device), label_s.to(device)
    
        pred1 = model(im_s)
        loss_seg = loss_fn(pred1, label_s)
        
        loss_seg.backward()
        optimizer.step()

        loss_data = loss_seg.data.item()
        pred = np.argmax(pred1.data.cpu().numpy()[0], axis=0)
        gt = label_s[0].data.cpu().numpy()
        train_acc.append(accuracy(pred,gt))
        train_loss.append(loss_data)

        log_file=osp.join(out_dir, 'training.log')
        if i_iter % 500 == 0:
            print('Train [{}/{} Source loss:{:.6f} acc:{:.4f}%]'.format(
                i_iter, args.num_steps, sum(train_loss)/len(train_loss),sum(train_acc)/len(train_acc)))

            message = 'Train  [{}/{} Source loss:{:.6f} acc:{:.4f}% ]'.format(
                i_iter, args.num_steps, sum(train_loss)/len(train_loss), sum(train_acc)/len(train_acc))

            with open(log_file, "a") as log_file:
                log_file.write('%s\n' % message) 

            train_loss=[]
            train_acc=[]

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('saving checkpoint.....')
            val(model, val_loader, args, out_dir, i_iter, device)
            torch.save(model.state_dict(), osp.join(checkpoint_dir,'iter{}.pth'.format(i_iter)))

        i_iter +=1

def val(model, dataloader, args, out_dir, i_iter, device):

    model.eval()
    pbar = tqdm(total=len(dataloader), desc="[Val]")
    conf_mat = metrics.ConfMatrix(args.num_classes)

    for i, batch in enumerate(dataloader):
        img, gt = batch['image'], batch['label']

        with torch.no_grad():
            pred = model(img.to(device))
            pred = np.argmax(pred.cpu().numpy(), axis=1)
        conf_mat.add_batch(gt, pred)
        pbar.update()

    pbar.set_description("[Val] OA: {:.2f}%".format(conf_mat.get_oa() * 100))
    pbar.close()
    print("OA\t", conf_mat.get_oa())
    print("AA\t", conf_mat.get_aa())
    print("mIoU\t", conf_mat.get_mIoU())

    log_file=osp.join(out_dir, 'training.log')
    message = 'Step: {}/{}\nOA: {:.6f}\nAA: {:.6f}\nmIoU: {:.6f}'.format(
            i_iter, args.num_steps, conf_mat.get_oa(), conf_mat.get_aa(), conf_mat.get_mIoU())
    with open(log_file, "a") as log_file:
        log_file.write('%s\n' % message) 

    model.train()
    return

if __name__ == '__main__':
    main()
