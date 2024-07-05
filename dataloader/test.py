import yaml
from attrdict import AttrDict
import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23/SSNS-Net-main')
from utils.show import show_one
from data_provider_pretraining import Train
import numpy as np
from PIL import Image
import random
import time
import os
""""""
seed = 555
np.random.seed(seed)
random.seed(seed)
cfg_file = 'pretraining_all.yaml'
with open('Miccai23/SSNS-Net-main/config/' + cfg_file, 'r') as f:
    cfg = AttrDict( yaml.safe_load(f) )

out_path = os.path.join('/braindat/lab/chenyd/code/Miccai23/SSNS-Net-main', 'data_temp')
if not os.path.exists(out_path):
    os.mkdir(out_path)
data = Train(cfg)
t = time.time()
for i in range(0, 20):
    t1 = time.time()
    tmp_data1, tmp_data2, gt = iter(data).__next__()
    print('single cost time: ', time.time()-t1)
    print('tmp_data1 shape: ', tmp_data1.shape, 'tmp_data2 shape: ', tmp_data2.shape, 'gt shape: ', gt.shape)
    tmp_data1 = np.squeeze(tmp_data1)
    tmp_data2 = np.squeeze(tmp_data2)
    gt = np.squeeze(gt)
    if cfg.MODEL.model_type == 'mala':
        tmp_data1 = tmp_data1[14:-14,106:-106,106:-106]
        tmp_data2 = tmp_data2[14:-14,106:-106,106:-106]
        gt = gt[14:-14,106:-106,106:-106]

    img_data1 = show_one(tmp_data1)
    img_data2 = show_one(tmp_data2)
    img_affs = show_one(gt)
    im_cat = np.concatenate([img_data1, img_data2, img_affs], axis=1)

    Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
print(time.time() - t)