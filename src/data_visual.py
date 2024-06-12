import h5py
from data.data_affinity import seg_to_aff
from data.data_segmentation import seg_widen_border, weight_binary_ratio
import numpy as np
from PIL import Image

# load raw data
f_raw = h5py.File('/h3cstore_ns/Backbones/data/wafer/wafer4_inputs.h5', 'r')
data = f_raw['main'][:]
f_raw.close()
        
# load labels
f_label = h5py.File('/h3cstore_ns/Backbones/data/wafer/wafer4_labels.h5', 'r')
label = f_label['main'][:]
f_label.close()

label = seg_widen_border(label, tsz_h=1)

gt_aff = seg_to_aff(label).astype(np.float32)

data_img = Image.fromarray((data[0,:]).astype(np.uint8))
aff_img = Image.fromarray((gt_aff[:, 0]*255).astype(np.uint8).transpose(1,2,0))

data_img.save("test_raw_img.png")
aff_img.save("test_aff_img.png")


print('done')

    # return data, label, gt_aff