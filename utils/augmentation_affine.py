import cv2
import math
import numpy as np

from utils import affine

class SegCVTransformRandomCropRotateScale(object):
    """
    Random crop with random scale.
    """
    def __init__(self, crop_size, crop_offset, rot_mag, max_scale, uniform_scale=True, constrain_rot_scale=True,
                 rng=None):
        if crop_offset is None:
            crop_offset = [0, 0]
        self.crop_size = tuple(crop_size)
        self.crop_size_arr = np.array(crop_size)
        self.crop_offset = np.array(crop_offset)
        self.rot_mag_rad = math.radians(rot_mag)
        self.log_max_scale = np.log(max_scale)
        self.uniform_scale = uniform_scale
        self.constrain_rot_scale = constrain_rot_scale
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    def transform_single(self, sample0):
        sample0 = sample0.copy()

        # Extract contents
        image = sample0['image_arr']

        # Choose scale and rotation
        if self.uniform_scale:
            scale_factor_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1,)))
            scale_factor_yx = np.repeat(scale_factor_yx, 2, axis=0)
        else:
            scale_factor_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2,)))
        rot_theta = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(1,))

        # Scale the crop size by the inverse of the scale
        sc_size = self.crop_size_arr / scale_factor_yx

        # Randomly choose centre
        img_size = np.array(image.shape[:2])
        extra = np.maximum(img_size - sc_size, 0.0)
        centre = extra * self.rng.uniform(0.0, 1.0, size=(2,)) + np.minimum(sc_size, img_size) * 0.5

        # Build affine transformation matrix
        local_xf = affine.cat_nx2x3(
            affine.translation_matrices(self.crop_size_arr[None, ::-1] * 0.5),
            affine.rotation_matrices(rot_theta),
            affine.scale_matrices(scale_factor_yx[None, ::-1]),
            affine.translation_matrices(-centre[None, ::-1]),
        )

        # Reflect the image
        # Use nearest neighbour sampling to stay consistent with labels, if labels present
        if 'labels_arr' in sample0:
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = self.rng.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])

        sample0['image_arr'] = cv2.warpAffine(image, local_xf[0], self.crop_size[::-1], flags=interpolation, borderValue=0, borderMode=cv2.BORDER_REFLECT_101)

        # Don't reflect labels and mask
        if 'labels_arr' in sample0:
            sample0['labels_arr'] = cv2.warpAffine(sample0['labels_arr'], local_xf[0], self.crop_size[::-1], flags=cv2.INTER_NEAREST, borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = cv2.warpAffine(sample0['mask_arr'], local_xf[0], self.crop_size[::-1], flags=interpolation, borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'xf_cv' in sample0:
            sample0['xf_cv'] = affine.cat_nx2x3(local_xf, sample0['xf_cv'][None, ...])[0]

        return sample0

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Choose scales and rotations
        if self.constrain_rot_scale:
            if self.uniform_scale:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1, 1)))
                scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=1)
            else:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(1, 2)))

            rot_thetas = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(1,))
            scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=0)
            rot_thetas = np.repeat(rot_thetas, 2, axis=0)
        else:
            if self.uniform_scale:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2, 1)))
                scale_factors_yx = np.repeat(scale_factors_yx, 2, axis=1)
            else:
                scale_factors_yx = np.exp(self.rng.uniform(-self.log_max_scale, self.log_max_scale, size=(2, 2)))
            rot_thetas = self.rng.uniform(-self.rot_mag_rad, self.rot_mag_rad, size=(2,))

        img_size = np.array(sample0['image_arr'].shape[:2])

        # Scale the crop size by the inverse of the scale
        sc_size = self.crop_size_arr / scale_factors_yx.min(axis=0)
        crop_centre_pos = np.minimum(sc_size, img_size) * 0.5

        # Randomly choose centres
        extra = np.maximum(img_size - sc_size, 0.0)
        centre0 = extra * self.rng.uniform(0.0, 1.0, size=(2,)) + crop_centre_pos
        offset1 = np.round(self.crop_offset * self.rng.uniform(-1.0, 1.0, size=(2,)))
        centre_xlat = np.stack([centre0, centre0], axis=0)
        offset1_xlat = np.stack([np.zeros((2,)), offset1], axis=0)

        # Build affine transformation matrices
        local_xfs = affine.cat_nx2x3(
            affine.translation_matrices(self.crop_size_arr[None, ::-1] * 0.5),
            affine.translation_matrices(offset1_xlat[:, ::-1]),
            affine.rotation_matrices(rot_thetas),
            affine.scale_matrices(scale_factors_yx[:, ::-1]),
            affine.translation_matrices(-centre_xlat[:, ::-1]),
        )

        # Use nearest neighbour sampling to stay consistent with labels, if labels present
        interpolation = cv2.INTER_NEAREST if 'labels_arr' in sample0 else cv2.INTER_LINEAR
        sample0['image_arr'] = cv2.warpAffine(sample0['image_arr'], local_xfs[0], self.crop_size[::-1], flags=interpolation,
                                          borderValue=0, borderMode=cv2.BORDER_REFLECT_101)
        sample1['image_arr'] = cv2.warpAffine(sample1['image_arr'], local_xfs[1], self.crop_size[::-1], flags=interpolation,
                                          borderValue=0, borderMode=cv2.BORDER_REFLECT_101)

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = cv2.warpAffine(sample0['labels_arr'], local_xfs[0], self.crop_size[::-1], flags=cv2.INTER_NEAREST,
                                               borderValue=0, borderMode=cv2.BORDER_CONSTANT)
            sample1['labels_arr'] = cv2.warpAffine(sample1['labels_arr'], local_xfs[1], self.crop_size[::-1], flags=cv2.INTER_NEAREST,
                                               borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = cv2.warpAffine(sample0['mask_arr'], local_xfs[0], self.crop_size[::-1], flags=interpolation,
                                             borderValue=0, borderMode=cv2.BORDER_CONSTANT)
            sample1['mask_arr'] = cv2.warpAffine(sample1['mask_arr'], local_xfs[1], self.crop_size[::-1], flags=interpolation,
                                             borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        if 'xf_cv' in sample0:
            xf01 = affine.cat_nx2x3(local_xfs, np.stack([sample0['xf_cv'], sample1['xf_cv']], axis=0))
            sample0['xf_cv'] = xf01[0]
            sample1['xf_cv'] = xf01[1]

        return sample0, sample1


class SegCVTransformRandomFlip(object):
    def __init__(self, hflip, vflip, hvflip, rng=None):
        self.hflip = hflip
        self.vflip = vflip
        self.hvflip = hvflip
        self.__rng = rng

    @property
    def rng(self):
        if self.__rng is None:
            self.__rng = np.random.RandomState()
        return self.__rng

    @staticmethod
    def flip_image(img, flip_xyd):
        if flip_xyd[0]:
            img = img[:, ::-1]
        if flip_xyd[1]:
            img = img[::-1, ...]
        if flip_xyd[2]:
            img = np.swapaxes(img, 0, 1)
        return img.copy()

    def transform_single(self, sample):
        sample = sample.copy()

        # Flip flags
        flip_flags_xyd = self.rng.binomial(1, 0.5, size=(3,)) != 0
        flip_flags_xyd = flip_flags_xyd & np.array([self.hflip, self.vflip, self.hvflip])

        sample['image_arr'] = self.flip_image(sample['image_arr'], flip_flags_xyd)

        if 'mask_arr' in sample:
            sample['mask_arr'] = self.flip_image(sample['mask_arr'], flip_flags_xyd)

        if 'labels_arr' in sample:
            sample['labels_arr'] = self.flip_image(sample['labels_arr'], flip_flags_xyd)

        if 'xf_cv' in sample:
            sample['xf_cv'] = affine.cat_nx2x3(
                affine.flip_xyd_matrices(flip_flags_xyd[None, ...], sample['image_arr'].shape[:2]),
                sample['xf_cv'][None, ...],
            )[0]

        return sample

    def transform_pair(self, sample0, sample1):
        sample0 = sample0.copy()
        sample1 = sample1.copy()

        # Flip flags
        flip_flags_xyd = self.rng.binomial(1, 0.5, size=(2, 3)) != 0
        flip_flags_xyd = flip_flags_xyd & np.array([[self.hflip, self.vflip, self.hvflip]])

        sample0['image_arr'] = self.flip_image(sample0['image_arr'], flip_flags_xyd[0])
        sample1['image_arr'] = self.flip_image(sample1['image_arr'], flip_flags_xyd[1])

        if 'mask_arr' in sample0:
            sample0['mask_arr'] = self.flip_image(sample0['mask_arr'], flip_flags_xyd[0])
            sample1['mask_arr'] = self.flip_image(sample1['mask_arr'], flip_flags_xyd[1])

        if 'labels_arr' in sample0:
            sample0['labels_arr'] = self.flip_image(sample0['labels_arr'], flip_flags_xyd[0])
            sample1['labels_arr'] = self.flip_image(sample1['labels_arr'], flip_flags_xyd[1])

        if 'xf_cv' in sample0:
            # False -> 1, True -> -1
            flip_scale_xy = flip_flags_xyd[:, :2] * -2 + 1
            # Negative scale factors need to be combined with a translation whose value is (image_size - 1)
            # Mask the translation with the flip flags to only apply it where flipping is done
            flip_xlat_xy = flip_flags_xyd[:, :2] * (np.array([sample0['image_arr'].shape[:2][::-1],
                                                              sample1['image_arr'].shape[:2][::-1]]).astype(float) - 1)

            hv_flip_xf = affine.identity_xf(2)
            hv_flip_xf[flip_flags_xyd[:, 2]] = hv_flip_xf[flip_flags_xyd[:, 2], ::-1, :]

            xf01 = np.stack([sample0['xf_cv'], sample1['xf_cv']], axis=0)
            xf01 = affine.cat_nx2x3(
                hv_flip_xf,
                affine.translation_matrices(flip_xlat_xy),
                affine.scale_matrices(flip_scale_xy),
                xf01,
            )
            sample0['xf_cv'] = xf01[0]
            sample1['xf_cv'] = xf01[1]

        return sample0, sample1

