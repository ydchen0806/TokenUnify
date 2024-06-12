import math
import torch
import numpy as np
from yaml.events import NodeEvent

class GenPseudo(object):
    def __init__(self, mode='threshold', 
                threshold=0.99, 
                proportion=0.20):
        super(GenPseudo, self).__init__()
        self.mode = mode
        self.threshold = threshold
        self.proportion = proportion

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        inputs = inputs.detach().clone()
        mask = torch.zeros_like(inputs)
        if self.mode == 'threshold':
            inputs[inputs > self.threshold] = 1
            mask[inputs == 1] = 1
            inputs[inputs < (1-self.threshold)] = 0
            mask[inputs == 0] = 1
            return inputs, mask
        else:
            num_classes = 2  # binary classification
            pseudo_lb = []
            masks = []
            batch_size = inputs.shape[0]
            for k in range(batch_size):
                output_affs = inputs[k]
                output_affs_0 = 1 - output_affs.clone()
                output_affs_1 = output_affs.clone()
                output_affs_all = torch.stack([output_affs_0, output_affs_1], dim=0)
                probmap_max, pred_label = torch.max(output_affs_all, dim=0)
                for idx_cls in range(num_classes):
                    out_div_all = []
                    for i in range(3):
                        pred_label_temp = pred_label[i]
                        probmap_max_temp = probmap_max[i]
                        probmap_max_cls_temp = probmap_max_temp[pred_label_temp == idx_cls]
                        if len(probmap_max_cls_temp) > 0:
                            # probmap_max_cls_temp = probmap_max_cls_temp.view(probmap_max_cls_temp.size(0), -1)
                            probmap_max_cls_temp = probmap_max_cls_temp[0:len(probmap_max_cls_temp)]
                            probmap_max_cls_temp, _ = torch.sort(probmap_max_cls_temp, descending=True)
                            len_cls = len(probmap_max_cls_temp)
                            thresh_len = int(math.floor(len_cls * self.proportion))
                            thresh_temp = probmap_max_cls_temp[thresh_len - 1]
                            out_div = torch.div(output_affs_all[idx_cls, i], thresh_temp)
                        else:
                            out_div = output_affs_all[idx_cls, i]
                        out_div_all.append(out_div)
                    out_div_all = torch.stack(out_div_all, dim=0)
                    output_affs_all[idx_cls] = out_div_all

                rw_probmap_max, pseudo_label = torch.max(output_affs_all, dim=0)
                mask = torch.zeros_like(rw_probmap_max)
                mask[rw_probmap_max>=1] = 1
                pseudo_lb.append(pseudo_label)
                masks.append(mask)
            pseudo_lb = torch.stack(pseudo_lb, dim=0)
            masks = torch.stack(masks, dim=0)

            return pseudo_lb, masks


if __name__ == "__main__":
    gen_pseudo = GenPseudo(mode='prop')
    pred = np.random.random((2,3,18,160,160)).astype(np.float32)
    pred = torch.tensor(pred).to('cuda:0')

    pseudo_lb, masks = gen_pseudo(pred)
