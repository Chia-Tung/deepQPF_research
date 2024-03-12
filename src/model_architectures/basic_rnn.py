import torch
from torch import nn


class BasicRNN(nn.Module):
    """Convoluctional RNN Block"""

    def __init__(
        self,
        num_filter,
        h_w,
        h2h_kernel=(3, 3),
        h2h_dilate=(1, 1),
        i2h_kernel=(3, 3),
        i2h_stride=(1, 1),
        i2h_pad=(1, 1),
        i2h_dilate=(1, 1),
        act_type=torch.tanh,
        prefix="BaseConvRNN",
    ):
        super().__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (
            self._h2h_kernel[1] % 2 == 1
        ), "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (
            h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
            h2h_dilate[1] * (h2h_kernel[1] - 1) // 2,
        )
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(h_w) == 2
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._height, self._width = h_w
        self._state_height = (
            self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h
        ) // self._i2h_stride[0] + 1
        self._state_width = (
            self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w
        ) // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0
