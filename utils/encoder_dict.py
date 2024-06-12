import torch

ENCODER_DICT = [
    'embed_in.0.weight',
    'embed_in.0.bias',
    'conv0.block1.0.weight',
    'conv0.block1.1.weight',
    'conv0.block1.1.bias',
    'conv0.block2.0.weight',
    'conv0.block2.1.weight',
    'conv0.block2.1.bias',
    'conv0.block2.3.weight',
    'conv0.block3.weight',
    'conv0.block3.bias',
    'conv1.block1.0.weight',
    'conv1.block1.1.weight',
    'conv1.block1.1.bias',
    'conv1.block2.0.weight',
    'conv1.block2.1.weight',
    'conv1.block2.1.bias',
    'conv1.block2.3.weight',
    'conv1.block3.weight',
    'conv1.block3.bias',
    'conv2.block1.0.weight',
    'conv2.block1.1.weight',
    'conv2.block1.1.bias',
    'conv2.block2.0.weight',
    'conv2.block2.1.weight',
    'conv2.block2.1.bias',
    'conv2.block2.3.weight',
    'conv2.block3.weight',
    'conv2.block3.bias',
    'conv3.block1.0.weight',
    'conv3.block1.1.weight',
    'conv3.block1.1.bias',
    'conv3.block2.0.weight',
    'conv3.block2.1.weight',
    'conv3.block2.1.bias',
    'conv3.block2.3.weight',
    'conv3.block3.weight',
    'conv3.block3.bias',
    'center.block1.0.weight',
    'center.block1.1.weight',
    'center.block1.1.bias',
    'center.block2.0.weight',
    'center.block2.1.weight',
    'center.block2.1.bias',
    'center.block2.3.weight',
    'center.block3.weight',
    'center.block3.bias'
]

ENCODER_DICT2 = [
    'embed_in',
    'conv0',
    'conv1',
    'conv2',
    'conv3',
    'center'
]

ENCODER_DECODER_DICT2 = [
    'embed_in',
    'conv0',
    'conv1',
    'conv2',
    'conv3',
    'center',
    'up0',
    'cat0',
    'conv4',
    'up1',
    'cat1',
    'conv5',
    'up2',
    'cat2',
    'conv6',
    'up3',
    'cat3',
    'conv7',
    'embed_out'
]

def freeze_layers(model, if_skip='False'):
    print('Freeze encoder!')
    for param in model.embed_in.parameters():
        param.requires_grad = False
    for param in model.conv0.parameters():
        param.requires_grad = False
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.conv3.parameters():
        param.requires_grad = False
    for param in model.center.parameters():
        param.requires_grad = False
    if if_skip == 'True':
        print('Freeze encoder and decoder!')
        for param in model.up0.parameters():
            param.requires_grad = False
        for param in model.cat0.parameters():
            param.requires_grad = False
        for param in model.conv4.parameters():
            param.requires_grad = False
        for param in model.up1.parameters():
            param.requires_grad = False
        for param in model.cat1.parameters():
            param.requires_grad = False
        for param in model.conv5.parameters():
            param.requires_grad = False
        for param in model.up2.parameters():
            param.requires_grad = False
        for param in model.cat2.parameters():
            param.requires_grad = False
        for param in model.conv6.parameters():
            param.requires_grad = False
        for param in model.up3.parameters():
            param.requires_grad = False
        for param in model.cat3.parameters():
            param.requires_grad = False
        for param in model.conv7.parameters():
            param.requires_grad = False
        for param in model.embed_out.parameters():
            param.requires_grad = False
        for param in model.out_put.parameters():
            param.requires_grad = False
    return model

def difflr_optimizer(model, lr_base=1e-4, lr_encoder=1e-5, if_skip='False'):
    print('Adjust the LR of encoder!')
    encoder_layers_param = []
    encoder_layers_param += list(map(id, model.embed_in.parameters()))
    encoder_layers_param += list(map(id, model.conv0.parameters()))
    encoder_layers_param += list(map(id, model.conv1.parameters()))
    encoder_layers_param += list(map(id, model.conv2.parameters()))
    encoder_layers_param += list(map(id, model.conv3.parameters()))
    encoder_layers_param += list(map(id, model.center.parameters()))
    if if_skip == 'True':
        print('Adjust the LR of encoder and decoder!')
        encoder_layers_param += list(map(id, model.up0.parameters()))
        encoder_layers_param += list(map(id, model.cat0.parameters()))
        encoder_layers_param += list(map(id, model.conv4.parameters()))
        encoder_layers_param += list(map(id, model.up1.parameters()))
        encoder_layers_param += list(map(id, model.cat1.parameters()))
        encoder_layers_param += list(map(id, model.conv5.parameters()))
        encoder_layers_param += list(map(id, model.up2.parameters()))
        encoder_layers_param += list(map(id, model.cat2.parameters()))
        encoder_layers_param += list(map(id, model.conv6.parameters()))
        encoder_layers_param += list(map(id, model.up3.parameters()))
        encoder_layers_param += list(map(id, model.cat3.parameters()))
        encoder_layers_param += list(map(id, model.conv7.parameters()))
        encoder_layers_param += list(map(id, model.embed_out.parameters()))
    encoder_param = filter(lambda p: id(p) in encoder_layers_param, model.parameters())
    decoder_param = filter(lambda p: id(p) not in encoder_layers_param, model.parameters())
    optimizer = torch.optim.Adam([{'params': encoder_param, 'lr': lr_encoder},
                                  {'params': decoder_param}], 
                                lr=lr_base, betas=(0.9, 0.999), eps=0.01, weight_decay=1e-6, amsgrad=True)
    return optimizer
