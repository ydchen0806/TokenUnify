import mahotas
import numpy as np
from scipy import ndimage

def randomlabel(segmentation):
    segmentation = segmentation.astype(np.uint32)
    uid = np.unique(segmentation)
    mid = int(uid.max()) + 1
    mapping = np.zeros(mid, dtype=segmentation.dtype)
    mapping[uid] = np.random.choice(len(uid), len(uid), replace=False).astype(segmentation.dtype)#(len(uid), dtype=segmentation.dtype)
    out = mapping[segmentation]
    out[segmentation==0] = 0
    return out

def watershed(affs, seed_method, use_mahotas_watershed=True):
    affs_xy = 1.0 - 0.5*(affs[1] + affs[2])
    depth  = affs_xy.shape[0]
    fragments = np.zeros_like(affs[0]).astype(np.uint64)
    next_id = 1
    for z in range(depth):
        seeds, num_seeds = get_seeds(affs_xy[z], next_id=next_id, method=seed_method)
        if use_mahotas_watershed:
            fragments[z] = mahotas.cwatershed(affs_xy[z], seeds)
        else:
            fragments[z] = ndimage.watershed_ift((255.0*affs_xy[z]).astype(np.uint8), seeds)
        next_id += num_seeds
    return fragments

def get_seeds(boundary, method='grid', next_id=1, seed_distance=10):
    if method == 'grid':
        height = boundary.shape[0]
        width  = boundary.shape[1]
        seed_positions = np.ogrid[0:height:seed_distance, 0:width:seed_distance]
        num_seeds_y = seed_positions[0].size
        num_seeds_x = seed_positions[1].size
        num_seeds = num_seeds_x*num_seeds_y
        seeds = np.zeros_like(boundary).astype(np.int32)
        seeds[seed_positions] = np.arange(next_id, next_id + num_seeds).reshape((num_seeds_y,num_seeds_x))

    if method == 'minima':
        minima = mahotas.regmin(boundary)
        seeds, num_seeds = mahotas.label(minima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    if method == 'maxima_distance':
        distance = mahotas.distance(boundary<0.5)
        maxima = mahotas.regmax(distance)
        seeds, num_seeds = mahotas.label(maxima)
        seeds += next_id
        seeds[seeds==next_id] = 0

    return seeds, num_seeds


def elf_watershed(affs):
    import elf.segmentation.watershed as ws
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    fragments = np.zeros_like(boundary_input, dtype='uint64')
    offset = 0
    for z in range(fragments.shape[0]):
        wsz, max_id = ws.distance_transform_watershed(boundary_input[z], threshold=.25, sigma_seeds=2.)
        wsz += offset
        offset += max_id
        fragments[z] = wsz
    return fragments

def relabel(seg):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-background samples
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1 # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]

def remove_small(seg, thres=100):
    sz = seg.shape
    seg = seg.reshape(-1)
    uid, uc = np.unique(seg, return_counts=True)
    seg[np.in1d(seg,uid[uc<thres])] = 0
    return seg.reshape(sz)
