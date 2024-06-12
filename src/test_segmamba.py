import torch
try:
    from tensorboardX import SummaryWriter
except:
    from torch.utils.tensorboard import SummaryWriter



from segmamba import SegMamba



model = SegMamba(in_chans=1, out_chans=3)
device = torch.device('cuda')

x = torch.randn(1, 1,16,160,160).to(device)
print(x.shape)