import numpy as np
from em_segLib.seg_malis import malis_init, malis_loss_weights_both
from em_segLib.seg_util import mknhood3d

def malis_loss(output_affs, test_label, seg):
    seg = seg.astype(np.uint64)
    conn_dims = np.array(output_affs.shape).astype(np.uint64)
    nhood_dims = np.array((3,3),dtype=np.uint64)
    nhood_data = mknhood3d(1).astype(np.int32).flatten()
    pre_ve, pre_prodDims, pre_nHood = malis_init(conn_dims, nhood_data, nhood_dims)
    weight = malis_loss_weights_both(seg.flatten(), conn_dims, nhood_data, nhood_dims, pre_ve, 
                    pre_prodDims, pre_nHood, output_affs.flatten(), test_label.flatten(), 0.5).reshape(conn_dims)
    malis = np.sum(weight * (output_affs - test_label) ** 2)
    return malis
