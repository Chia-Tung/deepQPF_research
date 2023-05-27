import torch.nn as nn
from typing import Dict, List

from src.model_architectures.basic_gru import BasicGRU

def make_layers(block: Dict[str, List[int]]):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(
                in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    model = nn.Sequential()
    for name, layer in layers:
        model.add_module(name, layer)
    return model

def get_encoder_params_GRU(input_channel_count, input_shape):
    print(f'[EncoderParams] channel_count:{input_channel_count} Shape:{input_shape}')
    msg = 'Input gets downsampled by a factor of 5, 3 and 2 sequentially'\
        'and subsequently it gets upsampled.'
    assert input_shape[0] % 30 == 0, msg
    assert input_shape[1] % 30 == 0, msg
    x, y = input_shape
    encoder_params = [
        [
            {'conv1_leaky_1': [input_channel_count, 8, 7, 5, 1]},
            {'conv2_leaky_1': [32, 128, 5, 3, 1]},
            {'conv3_leaky_1': [128, 128, 3, 2, 1]},
        ], 
        [
            BasicGRU(
                input_channel=8,
                num_filter=32,
                h_w=(x // 5, y // 5),
                zoneout=0.0,
                L=13,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(5, 5), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            BasicGRU(
                input_channel=128,
                num_filter=128,
                h_w=(x // 15, y // 15),
                zoneout=0.0,
                L=13,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(5, 5), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            BasicGRU(
                input_channel=128,
                num_filter=128,
                h_w=(x // 30, y // 30),
                zoneout=0.0,
                L=9,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(3, 3), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        ]
    ]
    return encoder_params

def get_forecaster_params_GRU():
    forecaster_params_GRU = [
        [
            {'deconv1_leaky_1': [128, 128, 4, 2, 1]},
            {'deconv2_leaky_1': [128, 32, 5, 3, 1]},
            {
                'deconv3_leaky_1': [32, 8, 7, 5, 1],
                'conv3_leaky_2': [8, 8, 3, 1, 1],
                'conv3_3': [8, 1, 1, 1, 0]
            },
        ],
        [
            BasicGRU(
                input_channel=128,
                num_filter=128,
                h_w=None,  # For forecaster, this variable is never used.
                zoneout=0.0,
                L=13,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(3, 3), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            BasicGRU(
                input_channel=128,
                num_filter=128,
                h_w=None,  # For forecaster, this variable is never used.
                zoneout=0.0,
                L=13,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(5, 5), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ),
            BasicGRU(
                input_channel=32,
                num_filter=32,
                h_w=None,  # For forecaster, this variable is never used.
                zoneout=0.0,
                L=9,
                i2h_kernel=(3, 3),
                i2h_stride=(1, 1),
                i2h_pad=(1, 1),
                h2h_kernel=(5, 5), # For Basic_GRU, this variable is never used.
                h2h_dilate=(1, 1), # For Basic_GRU, this variable is never used.
                act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        ]
    ]
    return forecaster_params_GRU

def get_aux_encoder_params(input_channel):
    """
    The output shape will be [B, 128, H//30, W//30]
    """
    print(f'[PoniEncoderParams] channel_count:{input_channel}')
    return [
        {'conv1_leaky_1': [input_channel, 8, 7, 5, 1]},
        {'conv2_leaky_1': [8, 32, 5, 3, 1]},
        {'conv3_leaky_1': [32, 128, 3, 2, 1]},
    ]