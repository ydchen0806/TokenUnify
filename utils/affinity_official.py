import numpy as np
from affogato.affinities import compute_multiscale_affinities, compute_affinities


# Copy from https://github.com/inferno-pytorch/neurofire/blob/master/neurofire/criteria/multi_scale_loss.py
class Downsampler(object):
    def __init__(self, scale_factor, ndim=None):
        assert isinstance(scale_factor, (list, int, tuple))
        if isinstance(scale_factor, (list, tuple)):
            assert all(isinstance(sf, int) for sf in scale_factor)
            if ndim is None:
                self.ndim = len(scale_factor)
            else:
                assert len(scale_factor) == ndim
                self.ndim = ndim
            self.scale_factor = scale_factor
        else:
            assert ndim is not None, "Cannot infer dimension from scalar downsample factor"
            self.ndim = ndim
            self.scale_factor = self.ndim * (scale_factor,)
        self.ds_slice = tuple(slice(None, None, sf) for sf in scale_factor)

    def __call__(self, input_):
        if input_.ndim > self.ndim:
            assert input_.ndim == self.ndim + 1, "%i, %i" % (input_.ndim, self.ndim)
            ds_slice = (slice(None),) + self.ds_slice
        else:
            ds_slice = self.ds_slice
        return input_[ds_slice]


# Copy from https://github.com/inferno-pytorch/neurofire/blob/master/neurofire/transform/affinities.py
class Segmentation2Affinities2or3D(object):
    def __init__(self, offsets, dtype='float32',
                 retain_mask=False, ignore_label=None,
                 retain_segmentation=False, segmentation_to_binary=False,
                 map_to_foreground=True, learn_ignore_transitions=False,
                 **super_kwargs):
        assert compute_affinities is not None,\
            "Couldn't find 'affogato' module, affinity calculation is not available"
        # assert pyu.is_listlike(offsets), "`offsets` must be a list or a tuple."
        super(Segmentation2Affinities2or3D, self).__init__(**super_kwargs)
        self.dim = len(offsets[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(off) == self.dim for off in offsets[1:])
        self.offsets = offsets
        self.dtype = dtype
        self.retain_mask = retain_mask
        self.ignore_label = ignore_label
        self.retain_segmentation = retain_segmentation
        self.segmentation_to_binary = segmentation_to_binary
        assert not (self.retain_segmentation and self.segmentation_to_binary),\
            "Currently not supported"
        self.map_to_foreground = map_to_foreground
        self.learn_ignore_transitions = learn_ignore_transitions

    def to_binary_segmentation(self, tensor):
        assert self.ignore_label != 0, "We assume 0 is background, not ignore label"
        if self.map_to_foreground:
            return (tensor == 0).astype(self.dtype)
        else:
            return (tensor != 0).astype(self.dtype)

    def include_ignore_transitions(self, affs, mask, seg):
        ignore_seg = (seg == self.ignore_label).astype(seg.dtype)
        ignore_transitions, invalid_mask = compute_affinities(ignore_seg, self.offsets)
        invalid_mask = np.logical_not(invalid_mask)
        # NOTE affinity convention returned by affogato:
        # transitions are marked by 0
        ignore_transitions = ignore_transitions == 0
        ignore_transitions[invalid_mask] = 0
        affs[ignore_transitions] = 0
        mask[ignore_transitions] = 1
        return affs, mask

    def input_function(self, tensor):
        # print("affs: in shape", tensor.shape)
        if self.ignore_label is not None:
            # output.shape = (C, Z, Y, X)
            output, mask = compute_affinities(tensor, self.offsets,
                                              ignore_label=self.ignore_label,
                                              have_ignore_label=True)
            if self.learn_ignore_transitions:
                output, mask = self.include_ignore_transitions(output, mask, tensor)
        else:
            output, mask = compute_affinities(tensor, self.offsets)

        # FIXME what does this do, need to refactor !
        # hack for platyneris data
        platy_hack = False
        if platy_hack:
            chan_mask = mask[1].astype('bool')
            output[0][chan_mask] = np.min(output[:2], axis=0)[chan_mask]

            chan_mask = mask[2].astype('bool')
            output[0][chan_mask] = np.minimum(output[0], output[2])[chan_mask]

        # Cast to be sure
        if not output.dtype == self.dtype:
            output = output.astype(self.dtype)
        #
        # print("affs: shape before binary", output.shape)
        if self.segmentation_to_binary:
            output = np.concatenate((self.to_binary_segmentation(tensor)[None],
                                     output), axis=0)
        # print("affs: shape after binary", output.shape)

        # print("affs: shape before mask", output.shape)
        # We might want to carry the mask along.
        # If this is the case, we insert it after the targets.
        if self.retain_mask:
            mask = mask.astype(self.dtype, copy=False)
            if self.segmentation_to_binary:
                if self.ignore_label is None:
                    additional_mask = np.ones((1,) + tensor.shape, dtype=self.dtype)
                else:
                    additional_mask = (tensor[None] != self.ignore_label).astype(self.dtype)
                mask = np.concatenate([additional_mask, mask], axis=0)
            output = np.concatenate((output, mask), axis=0)
        # print("affs: shape after mask", output.shape)

        # We might want to carry the segmentation along for validation.
        # If this is the case, we insert it before the targets.
        if self.retain_segmentation:
            # Add a channel axis to tensor to make it (C, Z, Y, X) before cating to output
            output = np.concatenate((tensor[None].astype(self.dtype, copy=False), output),
                                    axis=0)

        # print("affs: out shape", output.shape)
        return output


# Copy from https://github.com/inferno-pytorch/neurofire/blob/master/neurofire/transform/affinities.py
class Segmentation2MultiscaleAffinities(object):
    def __init__(self, block_shapes, dtype='float32', ignore_label=None,
                 retain_mask=False, retain_segmentation=False,
                 original_scale_offsets=None, **super_kwargs):
        super(Segmentation2MultiscaleAffinities, self).__init__(**super_kwargs)
        assert compute_multiscale_affinities is not None,\
            "Couldn't find 'affogato' module, affinity calculation is not available"
        # assert pyu.is_listlike(block_shapes)
        self.block_shapes = block_shapes
        self.dim = len(block_shapes[0])
        assert self.dim in (2, 3), str(self.dim)
        assert all(len(bs) == self.dim for bs in block_shapes[1:])

        self.dtype = dtype
        self.ignore_label = ignore_label
        self.retain_mask = retain_mask
        self.retain_segmentation = retain_segmentation
        self.original_scale_offsets = original_scale_offsets
        if self.retain_segmentation:
            self.downsamplers = [Downsampler(bs) for bs in self.block_shapes]

    def tensor_function(self, tensor):
        # for 2 d input, we need singleton input
        if self.dim == 2:
            assert tensor.shape[0] == 1
            tensor = tensor[0]

        outputs = []
        for ii, bs in enumerate(self.block_shapes):
            # if the block shape is all ones, we can compute normal affinities
            # with nearest neighbor offsets. This should yield the same result,
            # but should be more efficient.
            original_scale = all(s == 1 for s in bs)
            if original_scale:
                if self.original_scale_offsets is None:
                    offsets = [[0 if i != d else -1 for i in range(self.dim)]
                               for d in range(self.dim)]
                else:
                    offsets = self.original_scale_offsets
                output, mask = compute_affinities(tensor.squeeze().astype('uint64'), offsets,
                                                  ignore_label=0 if self.ignore_label is None else self.ignore_label,
                                                  have_ignore_label=False if self.ignore_label is None else True)
            else:
                output, mask = compute_multiscale_affinities(tensor.squeeze().astype('uint64'), bs,
                                                             ignore_label=0 if self.ignore_label is None
                                                             else self.ignore_label,
                                                             have_ignore_label=False if self.ignore_label is None
                                                             else True)

            # Cast to be sure
            if not output.dtype == self.dtype:
                output = output.astype(self.dtype)

            # We might want to carry the mask along.
            # If this is the case, we insert it after the targets.
            if self.retain_mask:
                output = np.concatenate((output, mask.astype(self.dtype, copy=False)), axis=0)
            # We might want to carry the segmentation along for validation.
            # If this is the case, we insert it before the targets for the original scale.
            if self.retain_segmentation:
                ds_target = self.downsamplers[ii](tensor.astype(self.dtype, copy=False))
                if ds_target.ndim != output.ndim:
                    assert ds_target.ndim == output.ndim - 1
                    ds_target = ds_target[None]
                output = np.concatenate((ds_target, output), axis=0)
            outputs.append(output)

        return outputs


def seg2affs(labels, offsets, dtype='float32',
            retain_mask=False, ignore_label=None,
            retain_segmentation=False, segmentation_to_binary=False,
            map_to_foreground=True, learn_ignore_transitions=False):
    instance = Segmentation2Affinities2or3D(offsets=offsets, dtype=dtype,
            retain_mask=retain_mask, ignore_label=ignore_label,
            retain_segmentation=retain_segmentation, segmentation_to_binary=segmentation_to_binary,
            map_to_foreground=map_to_foreground, learn_ignore_transitions=learn_ignore_transitions)
    affs = instance.input_function(labels)
    return affs


def seg2multiaffs(labels, block_shapes, dtype='float32', ignore_label=None,
                retain_mask=False, retain_segmentation=False,
                original_scale_offsets=None):
    instance = Segmentation2MultiscaleAffinities(block_shapes=block_shapes, dtype=dtype,
                ignore_label=ignore_label, retain_mask=retain_mask,
                retain_segmentation=retain_segmentation, original_scale_offsets=original_scale_offsets)
    affs = instance.tensor_function(labels)
    return affs
