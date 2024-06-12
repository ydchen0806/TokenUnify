import torch
import random
import numpy as np
import subprocess

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def center_crop(image, det_shape=[18, 160, 160]):
    # To prevent overflow
    image = np.pad(image, ((2,2),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image