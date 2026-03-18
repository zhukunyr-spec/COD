import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from lib.TDMLNet_Pvt import TDMLNet
from utils.dataloader import My_test_dataset 


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()F
parser.add_argument('--testsize', type=int, default=512, help='testing size default 352')
parser.add_argument('--pth_path', type=str, default='') 
opt = parser.parse_args()
for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:
    data_path = './'.format(_data_name)
    save_path = './'.format(opt.pth_path.split('/')[-2], _data_name)
    
    model = FMSDRNet(train_mode=False)
    model.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(opt.pth_path).items()})
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    print('root',image_root,gt_root)
    test_loader = My_test_dataset(image_root, gt_root, opt.testsize)
    print('****',test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        print('***name',name)
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        P = model(image)
        P[-1] = (torch.tanh(P[-1]) + 1.0) / 2.0
        
        res = F.upsample(P[-1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        cv2.imwrite(save_path+name,res*255)
