## FUNCTIONS: data augmentation used for "superhuman" network
## Noted that it is only applied to 3D datasets
## Written by Wei Huang
## 2020/10/12
## reference: https://github.com/donglaiw/EM-network/blob/master/em_net/data/augmentation.py

import cv2
import time
import math
import random
import torch
import numpy as np
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.filters import gaussian_filter

from utils.coordinate import Coordinate

def produce_simple_aug(data, rule):
    '''Routine data augmentation, including flipping in x-, y- and z-dimensions, 
        and transposing x- and y-dimensions, they have 2^4=16 combinations
    Args:
        data: numpy array, [Z, Y, X], ndim=3
        rule: numpy array, list or tuple, but len(rule) = 4, such as rule=[1,1,0,0]
    '''
    assert data.ndim == 3 and len(rule) == 4
    # z reflection.
    if rule[0]:
        data = data[::-1, :, :]
    # x reflection.
    if rule[1]:
        data = data[:, :, ::-1]
    # y reflection.
    if rule[2]:
        data = data[:, ::-1, :]
    # Transpose in xy.
    if rule[3]:
        data = data.transpose(0, 2, 1)
    return data

########################################################################
def create_identity_transformation(shape, subsample=1):
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s/subsample)) for s in shape)
    step_width = tuple(float(shape[d]-1)/(subsample_shape[d]-1)
                       if subsample_shape[d] > 1 else 1 for d in range(dims))

    axis_ranges = (
        np.arange(subsample_shape[d], dtype=np.float32)*step_width[d]
        for d in range(dims)
    )
    return np.array(np.meshgrid(*axis_ranges, indexing='ij'), dtype=np.float32)


def upscale_transformation(transformation,
                            output_shape,
                            interpolate_order=1):
    input_shape = transformation.shape[1:]
    dims = len(output_shape)
    scale = tuple(float(s)/c for s, c in zip(output_shape, input_shape))

    scaled = np.zeros((dims,)+output_shape, dtype=np.float32)
    for d in range(dims):
        zoom(transformation[d], zoom=scale,
             output=scaled[d], order=interpolate_order)
    return scaled


def create_elastic_transformation(shape,
                                    control_point_spacing=100,
                                    jitter_sigma=10.0,
                                    subsample=1):
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s/subsample)) for s in shape)

    try:
        spacing = tuple((d for d in control_point_spacing))
    except:
        spacing = (control_point_spacing,)*dims
    try:
        sigmas = [s for s in jitter_sigma]
    except:
        sigmas = [jitter_sigma]*dims

    control_points = tuple(
        max(1, int(round(float(shape[d])/spacing[d])))
        for d in range(len(shape))
    )

    # jitter control points
    control_point_offsets = np.zeros(
        (dims,) + control_points, dtype=np.float32)
    for d in range(dims):
        if sigmas[d] > 0:
            control_point_offsets[d] = np.random.normal(
                scale=sigmas[d], size=control_points)
    transform = upscale_transformation(control_point_offsets, subsample_shape, interpolate_order=3)
    return transform


def rotate(point, angle):
    res = np.array(point)
    res[0] = math.sin(angle)*point[1] + math.cos(angle)*point[0]
    res[1] = -math.sin(angle)*point[0] + math.cos(angle)*point[1]
    return res


def create_rotation_transformation(shape, angle, subsample=1):
    dims = len(shape)
    subsample_shape = tuple(max(1, int(s/subsample)) for s in shape)
    control_points = (2,)*dims

    # map control points to world coordinates
    control_point_scaling_factor = tuple(float(s-1) for s in shape)

    # rotate control points
    center = np.array([0.5*(d-1) for d in shape])

    control_point_offsets = np.zeros(
        (dims,) + control_points, dtype=np.float32)
    for control_point in np.ndindex(control_points):
        point = np.array(control_point)*control_point_scaling_factor
        center_offset = np.array(
            [p-c for c, p in zip(center, point)], dtype=np.float32)
        rotated_offset = np.array(center_offset)
        rotated_offset[-2:] = rotate(center_offset[-2:], angle)
        displacement = rotated_offset - center_offset
        control_point_offsets[(slice(None),) + control_point] += displacement
    return upscale_transformation(control_point_offsets, subsample_shape)


def random_offset(max_misalign):
    return Coordinate((0,) + tuple(max_misalign - random.randint(0, 2*int(max_misalign)) for d in range(2)))


def misalign(transformation, prob_slip, prob_shift, max_misalign):
    num_sections = transformation[0].shape[0]
    shifts = [Coordinate((0, 0, 0))]*num_sections
    # orginal
    # for z in range(num_sections):
    #     r = random.random()
    #     if r <= prob_slip:
    #         shifts[z] = random_offset(max_misalign)
    #     elif r <= prob_slip + prob_shift:
    #         offset = random_offset(max_misalign)
    #         for zp in range(z, num_sections):
    #             shifts[zp] += offset
    
    # written by Wei Huang
    if random.random() > 0.5:
        # slip type
        for z in range(1, num_sections):
            if random.random() <= prob_slip:
                shifts[z] = random_offset(max_misalign)
    else:
        # translation type
        for z in range(1, num_sections):
            if random.random() <= prob_shift:
                offset = random_offset(max_misalign)
                for zp in range(z, num_sections):
                    shifts[zp] = offset
                break

    for z in range(num_sections):
        transformation[1][z, :, :] += shifts[z][1]
        transformation[2][z, :, :] += shifts[z][2]
    return transformation

def apply_transformation(image,
                        transformation,
                        interpolate=True,
                        outside_value=0,
                        output=None):
    order = 1 if interpolate == True else 0
    output = image.dtype if output is None else output
    return map_coordinates(image,
                            transformation,
                            output=output,
                            order=order,
                            mode='constant',
                            cval=outside_value)

########################################################################


class SimpleAugment(object):
    def __init__(self, skip_ratio=0.5):
        '''Routine data augmentation, including flipping in x-, y- and z-dimensions, 
            and transposing x- and y-dimensions, they have 2^4=16 combinations
        Args:
            skip_ratio: Probability of execution
        '''
        super(SimpleAugment, self).__init__()
        self.ratio = skip_ratio
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        '''
        Args:
            inputs: list, such as [imgs, label, ...], imgs and label are numpy arrays with ndim=3
        '''
        skiprand = np.random.rand()
        if skiprand < self.ratio:
            rule = np.random.randint(2, size=4)
            for idx in range(len(inputs)):
                inputs[idx] = produce_simple_aug(inputs[idx], rule)
            return inputs
        else:
            return inputs


class RandomRotationAugment(object):
    def __init__(self, skip_ratio=0.5):
        '''Random rotation augmentation in x-y plane
        Args:
            skip_ratio: Probability of execution
        '''
        super(RandomRotationAugment, self).__init__()
        self.ratio = skip_ratio
    
    def __call__(self, inputs, mask=None):
        return self.forward(inputs, mask)
    
    def forward(self, inputs, mask=None):
        '''
        Args:
            inputs: list, such as [imgs, label, ...], imgs and label are numpy arrays with ndim=3
        '''
        skiprand = np.random.rand()
        if skiprand < self.ratio:
            angle = random.randint(0, 360-1)
            center = tuple(np.array(inputs.shape)[1:] // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
            for k in range(inputs.shape[0]):
                inputs[k] = cv2.warpAffine(inputs[k], rot_mat, inputs[k].shape, flags=cv2.INTER_LINEAR)
            if mask is not None:
                for k in range(mask.shape[0]):
                    mask[k] = cv2.warpAffine(mask[k], rot_mat, mask[k].shape, flags=cv2.INTER_NEAREST)
                return inputs, mask
            else:
                return inputs
        else:
            if mask is not None:
                return inputs, mask
            else:
                return inputs

class IntensityAugment(object):
    def __init__(self, mode='mix',
                        skip_ratio=0.5,
                        CONTRAST_FACTOR=0.1,
                        BRIGHTNESS_FACTOR=0.1):
        '''Image intensity augmentation, including adjusting contrast and brightness
        Args:
            mode: '2D', '3D' or 'mix' (contains '2D' and '3D')
            skip_ratio: Probability of execution
            CONTRAST_FACTOR: Contrast factor
            BRIGHTNESS_FACTOR : Brightness factor
        '''
        super(IntensityAugment, self).__init__()
        assert mode == '3D' or mode == '2D' or mode == 'mix'
        self.mode = mode
        self.ratio = skip_ratio
        self.CONTRAST_FACTOR = CONTRAST_FACTOR
        self.BRIGHTNESS_FACTOR = BRIGHTNESS_FACTOR
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        skiprand = np.random.rand()
        if skiprand < self.ratio:
            if self.mode == 'mix':
                # The probability of '2D' is more than '3D'
                threshold = 1 - (1 - self.ratio) / 2
                mode_ = '3D' if skiprand > threshold else '2D'
            else:
                mode_ = self.mode
            if mode_ == '2D':
                inputs = self.augment2D(inputs)
            elif mode_ == '3D':
                inputs = self.augment3D(inputs)
            return inputs
        else:
            return inputs
    
    def augment2D(self, imgs):
        for z in range(imgs.shape[-3]):
            img = imgs[z, :, :]
            img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            img = np.clip(img, 0, 1)
            img **= 2.0**(np.random.rand()*2 - 1)
            imgs[z, :, :] = img
        return imgs
    
    def augment3D(self, imgs):
        imgs *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
        imgs += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
        imgs = np.clip(imgs, 0, 1)
        imgs **= 2.0**(np.random.rand()*2 - 1)
        return imgs


class ElasticAugment(object):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.
    Args:
        control_point_spacing (``tuple`` of ``int``):
            Distance between control points for the elastic deformation, in
            voxels per dimension.
        jitter_sigma (``tuple`` of ``float``):
            Standard deviation of control point jitter distribution, in voxels
            per dimension.
        rotation_interval (``tuple`` of two ``floats``):
            Interval to randomly sample rotation angles from (0, 2PI).
        prob_slip (``float``):
            Probability of a section to "slip", i.e., be independently moved in
            x-y.
        prob_shift (``float``):
            Probability of a section and all following sections to move in x-y.
        max_misalign (``int``):
            Maximal voxels to shift in x and y. Samples will be drawn
            uniformly. Used if ``prob_slip + prob_shift`` > 0.
        subsample (``int``):
            Instead of creating an elastic transformation on the full
            resolution, create one subsampled by the given factor, and linearly
            interpolate to obtain the full resolution transformation. This can
            significantly speed up this node, at the expense of having visible
            piecewise linear deformations for large factors. Usually, a factor
            of 4 can savely by used without noticable changes. However, the
            default is 1 (i.e., no subsampling).
    '''
    def __init__(
            self,
            control_point_spacing=[4, 40, 40],
            jitter_sigma=[0,0,0],   # recommend: [0, 2, 2]
            rotation_interval=[0,0],
            prob_slip=0,   # recommend: 0.05
            prob_shift=0,   # recommend: 0.05
            max_misalign=0,   # 17 in superhuman
            subsample=1,
            padding=None,
            skip_ratio=0.5):   # recommend: 10
        super(ElasticAugment, self).__init__()

        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.subsample = subsample
        self.padding = padding
        self.ratio = skip_ratio

    def create_transformation(self, target_shape):

        transformation = create_identity_transformation(
            target_shape,
            subsample=self.subsample)
        # shape: channel,d,w,h

        # elastic  ##cost time##
        if sum(self.jitter_sigma) > 0 and np.random.rand() < self.ratio:
            transformation += create_elastic_transformation(
                target_shape,
                self.control_point_spacing,
                self.jitter_sigma,
                subsample=self.subsample)

        # rotation = random.random()*self.rotation_max_amount + self.rotation_start
        # if rotation != 0:
        #     transformation += create_rotation_transformation(
        #         target_shape,
        #         rotation,
        #         subsample=self.subsample)

        # if self.subsample > 1:
        #     transformation = upscale_transformation(
        #         transformation,
        #         tuple(target_shape))

        if self.prob_slip + self.prob_shift > 0 and np.random.rand() < self.ratio:
            misalign(transformation, self.prob_slip,
                     self.prob_shift, self.max_misalign)

        return transformation

    def __call__(self, imgs, mask):
        return self.forward(imgs, mask)

    def forward(self, imgs, mask):
        '''Args:
            imgs: numpy array, [Z, Y, Z], it always is float and 0~1
            mask: numpy array, [Z, Y, Z], it always is uint16
        '''
        if self.padding is not None:
            imgs = np.pad(imgs, ((0,0), \
                                (self.padding,self.padding), \
                                (self.padding,self.padding)), mode='reflect')
            mask = np.pad(mask, ((0,0), \
                                (self.padding,self.padding), \
                                (self.padding,self.padding)), mode='reflect')
        transform = self.create_transformation(imgs.shape)
        img_transform = apply_transformation(imgs,
                                         transform,
                                         interpolate=False,
                                         outside_value=0,  # imgs.dtype.type(-1)
                                         output=np.zeros(imgs.shape, dtype=np.float32))
        seg_transform = apply_transformation(mask,
                                         transform,
                                         interpolate=False,
                                         outside_value=0,  # mask.dtype.type(-1)
                                         output=np.zeros(mask.shape, dtype=np.uint16))  # dtype=np.float32
        # seg_transform[seg_transform < 0] = 0
        # seg_transform[seg_transform > 60000] = 0
        if self.padding is not None and self.padding != 0:
            img_transform = img_transform[:, self.padding:-self.padding, self.padding:-self.padding]
            seg_transform = seg_transform[:, self.padding:-self.padding, self.padding:-self.padding]
        return img_transform, seg_transform


class MissingAugment(object):
    '''Missing section augmentation
    Args:
        filling: the way of filling, 'zero' or 'random'
        mode: 'mix', 'fully' or 'partially'
        skip_ratio: Probability of execution
        miss_ratio: Probability of missing
    '''
    def __init__(self, filling='zero', mode='mix', skip_ratio=0.5, miss_ratio=0.1):
        super(MissingAugment, self).__init__()
        self.filling = filling
        self.mode = mode
        self.ratio = skip_ratio
        self.miss_ratio = miss_ratio
    
    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        skiprand = np.random.rand()
        if skiprand < self.ratio:
            if self.mode == 'mix':
                r = np.random.rand()
                mode_ = 'fully' if r < 0.5 else 'partially'
            else:
                mode_ = self.mode
            if mode_ == 'fully':
                imgs = self.augment_fully(imgs)
            elif mode_ == 'partially':
                imgs = self.augment_partially(imgs)
            return imgs
        else:
            return imgs
    
    def augment_fully(self, imgs):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.miss_ratio:
                if self.filling == 'zero':
                    imgs[i] = 0
                elif self.filling == 'random':
                    imgs[i] = np.random.rand(h, w)
        return imgs
    
    def augment_partially(self, imgs, size_ratio=0.3):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.miss_ratio:
                # randomly generate an area
                sub_h = random.randint(int(h*size_ratio), int(h*(1-size_ratio)))
                sub_w = random.randint(int(w*size_ratio), int(w*(1-size_ratio)))
                start_h = random.randint(0, h - sub_h - 1)
                start_w = random.randint(0, w - sub_w - 1)
                if self.filling == 'zero':
                    imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = 0
                elif self.filling == 'random':
                    imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = np.random.rand(sub_h, sub_w)
        return imgs


class BlurAugment(object):
    '''Out-of-focus (Blur) section augmentation
    Args:
        mode: 'mix', 'fully' or 'partially'
        skip_ratio: Probability of execution
        blur_ratio: Probability of blur
    '''
    def __init__(self, mode='mix', skip_ratio=0.5, blur_ratio=0.1):
        super(BlurAugment, self).__init__()
        self.mode = mode
        self.ratio = skip_ratio
        self.blur_ratio = blur_ratio
    
    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        skiprand = np.random.rand()
        if skiprand < self.ratio:
            if self.mode == 'mix':
                r = np.random.rand()
                mode_ = 'fully' if r < 0.5 else 'partially'
            else:
                mode_ = self.mode
            if mode_ == 'fully':
                imgs = self.augment_fully(imgs)
            elif mode_ == 'partially':
                imgs = self.augment_partially(imgs)
            return imgs
        else:
            return imgs
    
    def augment_fully(self, imgs):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.blur_ratio:
                sigma = np.random.uniform(0, 5)
                imgs[i] = gaussian_filter(imgs[i], sigma)
        return imgs
    
    def augment_partially(self, imgs, size_ratio=0.3):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.blur_ratio:
                # randomly generate an area
                sub_h = random.randint(int(h*size_ratio), int(h*(1-size_ratio)))
                sub_w = random.randint(int(w*size_ratio), int(w*(1-size_ratio)))
                start_h = random.randint(0, h - sub_h - 1)
                start_w = random.randint(0, w - sub_w - 1)
                sigma = np.random.uniform(0, 5)
                imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = \
                    gaussian_filter(imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w], sigma)
        return imgs


def show(img3d):
    # only used for image with shape [18, 160, 160]
    row = 4
    column = 5
    num = 18
    size = 160
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all


def show_lb(img3d):
    # only used for image with shape [18, 160, 160]
    row = 4
    column = 5
    num = 18
    size = 160
    ids = np.unique(img3d)
    color_pred = np.zeros([num, size, size, 3], dtype=np.uint8)
    idx = np.searchsorted(ids, img3d)
    for i in range(3):
        color_val = np.random.randint(0, 255, ids.shape)
        if ids[0] == 0:
            color_val[0] = 0
        color_pred[:,:,:,i] = color_val[idx]

    img_all = np.zeros((size*row, size*column, 3), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like((size, size, 3), dtype=np.uint8)
            else:
                img = color_pred[index]
            img_all[i*size:(i+1)*size, j*size:(j+1)*size, :] = img
    return img_all


def elastic_deform_3d_cuda(image_in, 
                            label_in, 
                            prob, 
                            random_state=None, 
                            padding=20, 
                            alpha=(10,50), 
                            sigma=10, 
                            device='cuda:0'):
    """Elastic deformation of image_ins as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
    """	
    # skip
    if random.uniform(0, 1) > prob:
        return image_in, label_in
    
    if padding is not None:
        image_in = np.pad(image_in, ((0,0), \
                    (padding, padding), \
                    (padding, padding)), mode='reflect')
        label_in = np.pad(label_in, ((0,0), \
                    (padding, padding), \
                    (padding, padding)), mode='reflect')
    
    alpha = np.random.uniform(alpha[0], alpha[1])
    if random_state is None:
        random_state = np.random.RandomState(None)
    
    shape = image_in.shape
    
    #rdx = torch.Tensor(random_state.rand(*shape) * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
    #rdy = torch.Tensor(random_state.rand(*shape) * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
    #rdz = torch.Tensor(random_state.rand(*shape) * 2 - 1).unsqueeze(0).unsqueeze(0).to(self.device)
    #dx = self.gaussian_filter(rdx) * alpha
    #dy = self.gaussian_filter(rdy) * alpha
    #dz = self.gaussian_filter(rdz) * alpha
    
    #dx = np.squeeze(dx.data.cpu().numpy())
    #dy = np.squeeze(dy.data.cpu().numpy())
    #dz = np.squeeze(dz.data.cpu().numpy())
    
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, order=0, mode='constant', cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, order=0, mode='constant', cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, order=0, mode='constant', cval=0) * alpha
    
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    def_grid_x = grid_x + dx
    def_grid_y = grid_y + dy
    def_grid_z = grid_z + dz
    
    gx = 2.0*def_grid_x/(shape[2]-1)-1.
    gy = 2.0*def_grid_y/(shape[1]-1)-1.
    gz = 2.0*def_grid_z/(shape[0]-1)-1.
    
    #indices = np.reshape(def_grid_y, (-1, 1)), np.reshape(def_grid_x, (-1, 1)), np.reshape(def_grid_z, (-1, 1))
    #out = map_coordinates(image_in, indices, order=1).reshape(shape)
    
    # torch_grid = torch.Tensor(np.stack((gx,gy,gz),3)).unsqueeze(0).to(device)
    torch_grid = torch.Tensor(np.stack((gz,gx,gy),3)).unsqueeze(0).to(device)
    torch_im = torch.Tensor(np.expand_dims(np.expand_dims(image_in, axis=0), axis=0).copy()).to(device)
    torch_lb = torch.Tensor(np.expand_dims(label_in, axis=0).copy()).to(device)
    with torch.no_grad():
        torch_im_out = torch.nn.functional.grid_sample(torch_im, torch_grid, mode='bilinear', padding_mode='zeros')
        torch_lb_out = torch.nn.functional.grid_sample(torch_lb, torch_grid, mode='bilinear', padding_mode='zeros')
    
    image_out = np.squeeze(torch_im_out.data.cpu().numpy()).astype(np.uint8)
    label_out = np.squeeze(torch_lb_out.data.cpu().numpy()).astype(np.uint8)
    
    if padding is not None and padding != 0:
        image_out = image_out[:, padding:-padding, padding:-padding]
        label_out = label_out[:, padding:-padding, padding:-padding]
    return image_out, label_out

if __name__ == "__main__":
    import os
    import cv2
    import h5py
    
    input_vol = '../data/snemi3d/train-input.h5'
    f = h5py.File(input_vol, 'r')
    raw = f['main'][:]
    f.close()

    input_vol = '../data/snemi3d/train-labels.h5'
    f = h5py.File(input_vol, 'r')
    lbs = f['main'][:]
    f.close()

    out = './debug_img'
    raw = raw.astype(np.float32) / 255.0
    vol = raw[0:18, 0:160, 0:160]
    lb = lbs[0:18, 0:160, 0:160]
    # vol_img = show_lb(lb)
    # cv2.imwrite(os.path.join(out, 'raw.png'), vol_img)

    ##################################################
    # Data_aug = ElasticAugment(jitter_sigma=[0,2,2],
    #                         prob_slip=0.5,
    #                         prob_shift=0.5,
    #                         max_misalign=17,
    #                         padding=20)
    # print('min=%d, max=%d' % (np.min(lb), np.max(lb)))
    ##################################################
    # Data_aug = MissingAugment(filling='random')
    ##################################################
    Data_aug = BlurAugment(blur_ratio=0.1)
    for i in range(20):
        # vol_aug, lb_aug = Data_aug(vol.copy(), lb.copy())
        # print('min=%d, max=%d' % (np.min(lb_aug), np.max(lb_aug)))
        # print(lb_aug.dtype)
        # vol_img = show_lb(lb_aug)
        vol_aug = Data_aug(vol.copy())
        vol_img = show(vol_aug)
        
        cv2.imwrite(os.path.join(out, 'raw_aug'+str(i)+'.png'), vol_img)
    print('Done')