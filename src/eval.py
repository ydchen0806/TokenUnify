import numpy as np
import waterz
import os
from utils.fragment import watershed, randomlabel, relabel
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.lmc import mc_baseline
from PIL import Image

# import yaml
# from provider_valid import Provider_valid
# from attrdict import AttrDict
# cfg_file = 'seg_all_3d_ac4_data80'
# with open('/h3cstore_ns/hyshi/configs/' + cfg_file + '.yaml', 'r') as f:
#     cfg = AttrDict(yaml.safe_load(f))
# valid_provider = Provider_valid(cfg, test_split=cfg.DATA.test_split)
# valid_data = valid_provider.get_raw_data()
# print(valid_data.shape)
# np.save('/h3cstore_ns/hyshi/InferenceWafer4/valid_data.npy', valid_data)
# valid_label = valid_provider.get_gt_lb()
# print(valid_label.shape)
# np.save('/h3cstore_ns/hyshi/valid_label.npy', valid_label)


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

# input_dir = '/h3cstore_ns/hyshi/InferenceWafer4_monai'
# output_files = '/h3cstore_ns/hyshi/InferenceWafer4_monai_10.txt'

# input_dir = '/h3cstore_ns/hyshi/InferenceWafer36_2_monai'
# output_files = '/h3cstore_ns/hyshi/InferenceWafer36_2_monai_0520.txt'

# input_dir = '/h3cstore_ns/hyshi/InferenceAC3_monai'
# output_files = '/h3cstore_ns/hyshi/InferenceAC3_monai.txt'

# input_dir = '/h3cstore_ns/hyshi/InferenceAC3_monai'
# output_files = '/h3cstore_ns/hyshi/InferenceAC3_monai_0520.txt'

# input_dir = '/h3cstore_ns/hyshi/Visual_wafer4'
# output_files = '/h3cstore_ns/hyshi/Visual_wafer4.txt'

# input_dir = '/h3cstore_ns/hyshi/Visual_wafer36_2'
# output_files = '/h3cstore_ns/hyshi/Visual_wafer36_2.txt'

input_dir = '//h3cstore_ns/hyshi/wafer4_errorbar/unetr_MAE'
output_files = '/h3cstore_ns/hyshi/errorbar_unetr_MAE.txt'

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_wafer4.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_wafer4.npy')

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_ac3.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_ac3.npy')

# valid_data = np.load('/h3cstore_ns/hyshi/valid_data_wafer36_2.npy')
# valid_label = np.load('/h3cstore_ns/hyshi/valid_label_wafer36_2.npy')

file_names = os.listdir(input_dir)
with open(output_files, 'w', encoding='utf-8') as outfile:
    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)

        if os.path.isfile(file_path):
            print(f'load {file_name} data...')
            data_zip = np.load(file_path)
            pred_affs = data_zip['pred_affs']
            gt_seg = data_zip['gt_seg']
            gt_affs = data_zip['gt_affs']

            # print('Waterz Segmentation...')
            # fragments = watershed(pred_affs, 'maxima_distance')
            # sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
            # # sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
            # seg_waterz = list(waterz.agglomerate(pred_affs, [0.50],
            #                                     fragments=fragments,
            #                                     scoring_function=sf,
            #                                     discretize_queue=256))[0]
            # seg_waterz = relabel(seg_waterz).astype(np.uint64)
            # arand_waterz = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
            # voi_split_waterz, voi_merge_waterz = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
            # voi_sum_waterz = voi_split_waterz + voi_merge_waterz

            print('LMC Segmentation...')
            seg_lmc = mc_baseline(pred_affs)
            arand_lmc = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
            voi_split_lmc, voi_merge_lmc = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
            voi_sum_lmc = voi_split_lmc + voi_merge_lmc

            print('Write the results...')
            # print('VOIm-waterz=%.6f, VOIs-waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
    # (voi_merge_waterz, voi_split_waterz, voi_sum_waterz, arand_waterz, voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc), flush=True)
            print('VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
            (voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc), flush=True)
            outfile.write(f'{file_name}: \n')
            # outfile.write('VOIm-waterz=%.6f, VOIs-waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f\n' % \
        # (voi_merge_waterz, voi_split_waterz, voi_sum_waterz, arand_waterz, voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc))
            outfile.write('VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f\n' % \
        (voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc))
            
            # print('Visualize...')
            # waterz_seg_color = draw_fragments_3d(seg_waterz)
            # label_color = draw_fragments_3d(valid_label)
            # zero_positions = (label_color == 0)
            # waterz_seg_color[zero_positions] = 0
            # label_img = Image.fromarray(label_color[3].astype(np.uint8))
            # waterz_img = Image.fromarray(waterz_seg_color[3].astype(np.uint8))
            # h, w = label_img.size
            # white_line = Image.new('RGB', (w, 16), (255, 255, 255))
            # pred_aff_img = Image.fromarray((pred_affs[:,3]*255).astype(np.uint8).transpose(1,2,0))
            # data_img = Image.fromarray((valid_data[3]*255).astype(np.uint8))
            # affs_img = Image.fromarray((gt_affs[:,3]*255).astype(np.uint8).transpose(1,2,0))
            # visual_img = Image.new('RGB', (w*5 + 16*4, h), (255, 255, 255))
            # visual_img.paste(label_img, (0, 0))
            # visual_img.paste(waterz_img, (w+16, 0))
            # visual_img.paste(white_line, (w*2+16, 0))
            # visual_img.paste(pred_aff_img, (w*2+16*2, 0))
            # visual_img.paste(affs_img, (w*3+16*3, 0))
            # visual_img.paste(white_line, (0, h))
            # visual_img.paste(data_img, (w*4+16*4, 0))
            # # visual_img.save('/h3cstore_ns/hyshi/InferenceWafer4_monai_visual' + f'/{file_name}.png')
            # # visual_img.save('/h3cstore_ns/hyshi/InferenceWafer36_2_monai_visual' + f'/{file_name}.png')
            # # visual_img.save('/h3cstore_ns/hyshi/InferenceAC3_visual/0520' + f'/{file_name}.png')
            # # visual_img.save('/h3cstore_ns/hyshi/InferenceWafer36_2_monai_visual/0520' + f'/{file_name}.png')
            # # visual_img.save('/h3cstore_ns/hyshi/Visual_wafer4_10' + f'/{file_name}.png')
            # visual_img.save('/h3cstore_ns/hyshi/Visual_wafer36_2_3' + f'/{file_name}.png')
            print(f'{file_name} have done.')

        else:
            print('No such file.')




# print('VOIm-waterz=%.6f, VOIs-waterz=%.6f, VOI-waterz=%.6f, ARAND-waterz=%.6f, VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
#     (voi_merge_waterz, voi_split_waterz, voi_sum_waterz, arand_waterz, voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc), flush=True)

# print('VOIm-lmc=%.6f, VOIs-lmc=%.6f, VOI-lmc=%.6f, ARAND-lmc=%.6f' % \
#     (voi_merge_lmc, voi_split_lmc, voi_sum_lmc, arand_lmc), flush=True)