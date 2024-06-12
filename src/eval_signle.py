import numpy as np
import waterz
import os
from utils.fragment import watershed, randomlabel, relabel
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.lmc import mc_baseline
from PIL import Image

def draw_fragments_3d(pred):
    d,m,n = pred.shape
    ids = np.unique(pred)
    size = len(ids)
    print("the neurons number of pred is %d" % size)
    color_pred = np.zeros([d, m, n, 3])
    idx = np.searchsorted(ids, pred)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,:,i] = color_val[idx]
    color_pred = color_pred
    return color_pred

print(f'load data...')
data_zip = np.load('/h3cstore_ns/hyshi/wafer4_errorbar/unetr_MAE/unetr_MAE_720.npz')
pred_affs = data_zip['pred_affs']
gt_seg = data_zip['gt_seg']
gt_affs = data_zip['gt_affs']

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label.npy')

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_wafer4.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_wafer4.npy')

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_ac3.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_ac3.npy')

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_wafer36_2.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_wafer36_2.npy')

print('Waterz Segmentation...')
fragments = watershed(pred_affs, 'maxima_distance')
sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
# sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
seg_waterz = list(waterz.agglomerate(pred_affs, [0.50],
                                    fragments=fragments,
                                    scoring_function=sf,
                                    discretize_queue=256))[0]
seg_waterz = relabel(seg_waterz).astype(np.uint64)
arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
voi_split_waterz, voi_merge_waterz = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
voi_sum_waterz = voi_split_waterz + voi_merge_waterz

print('LMC Segmentation...')
seg_lmc = mc_baseline(pred_affs)
arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
voi_split_lmc, voi_merge_lmc = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
voi_sum_lmc = voi_split_lmc + voi_merge_lmc

# print('Write the results...')
# outfile.write(f'{file_name}: \n')
# outfile.write('VOIm-waterz=%.6f, VOIs-waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f\n' % \
# (voi_merge_waterz, voi_split_waterz, voi_sum_waterz, arand_waterz, voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc))

# print('Visualize...')
# waterz_seg_color = draw_fragments_3d(seg_waterz)
# label_color = draw_fragments_3d(valid_label)
# zero_positions = (label_color == 0)
# waterz_seg_color[zero_positions] = 0
# label_img = Image.fromarray(label_color[0].astype(np.uint8))
# waterz_img = Image.fromarray(waterz_seg_color[0].astype(np.uint8))
# h, w = label_img.size
# white_line = Image.new('RGB', (w, 16), (255, 255, 255))
# pred_aff_img = Image.fromarray((pred_affs[:,0]*255).astype(np.uint8).transpose(1,2,0))
# data_img = Image.fromarray((valid_data[0]*255).astype(np.uint8))
# affs_img = Image.fromarray((gt_affs[:,0]*255).astype(np.uint8).transpose(1,2,0))
# visual_img = Image.new('RGB', (w*5 + 16*4, h), (255, 255, 255))
# visual_img.paste(label_img, (0, 0))
# visual_img.paste(waterz_img, (w+16, 0))
# visual_img.paste(white_line, (w*2+16, 0))
# visual_img.paste(pred_aff_img, (w*2+16*2, 0))
# visual_img.paste(affs_img, (w*3+16*3, 0))
# visual_img.paste(white_line, (0, h))
# visual_img.paste(data_img, (w*4+16*4, 0))
# # visual_img.save('/h3cstore_ns/hyshi/InferenceWafer36_2_monai_visual/0520' + '/PEA_random_25000.png')
# visual_img.save('/h3cstore_ns/hyshi/Visual_wafer4_result' + '/mamba3_ar11_1150test.png')

print('VOIm-waterz=%.6f, VOIs-waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
    (voi_merge_waterz, voi_split_waterz, voi_sum_waterz, arand_waterz, voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc), flush=True)