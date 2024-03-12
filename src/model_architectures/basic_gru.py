import torch
from torch import nn

from src.model_architectures.basic_rnn import BasicRNN


class BasicGRU(BasicRNN):
    def __init__(
        self,
        input_channel,
        num_filter,
        h_w,
        zoneout=0.0,
        L=5,
        i2h_kernel=(3, 3),
        i2h_stride=(1, 1),
        i2h_pad=(1, 1),
        h2h_kernel=(5, 5),
        h2h_dilate=(1, 1),
        act_type=nn.LeakyReLU(negative_slope=0.2, inplace=True),
    ):
        super().__init__(
            num_filter=num_filter,
            h_w=h_w,
            h2h_kernel=h2h_kernel,
            h2h_dilate=h2h_dilate,
            i2h_kernel=i2h_kernel,
            i2h_pad=i2h_pad,
            i2h_stride=i2h_stride,
            act_type=act_type,
            prefix="SimpleGRU",
        )
        self._L = L
        self._zoneout = zoneout

        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(
            in_channels=input_channel,
            out_channels=self._num_filter * 3,
            kernel_size=self._i2h_kernel,
            stride=self._i2h_stride,
            padding=self._i2h_pad,
            dilation=self._i2h_dilate,
        )

        # NOTE: Parameters from `get_encoder_params_GRU` and
        #       `get_forecaster_params_GRU` are not used here.
        self.h2h = nn.Conv2d(
            in_channels=self._num_filter,
            out_channels=self._num_filter * 3,
            kernel_size=(5, 5),
            stride=1,
            padding=(2, 2),
            dilation=(1, 1),
        )

    # shape of inputs: [S, B, C, H, W]
    def forward(
        self,
        inputs=None,
        states=None,
        seq_len=None,
    ):
        """
        Args:
            inputs: Shape of [S, B, C1, H1, W1]
            states: Shape of [B, C2, H1, W1]
            seq_len: Same as the 1st dim of inputs
        Returns:
            outputs: Shape of [S, B, C2, H1, W1]
            next_h: Shape of [B, C2, H1, W1]
        """
        assert seq_len is not None
        if states is None:
            # NOTE: The `state_height` and `state_width` are the shape after
            #       passing through the cnn block of `i2h`
            states = torch.zeros(
                (
                    inputs.size(1),
                    self._num_filter,
                    self._state_height,
                    self._state_width,
                ),
                dtype=torch.float,
            )
            states = states.type_as(inputs)

        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)

        prev_h = states
        outputs = []
        for i in range(seq_len):
            h2h = self.h2h(prev_h)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if inputs is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(
                    i2h_slice[2][i, ...] + reset_gate * h2h_slice[2]
                )
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])

            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            outputs.append(next_h)
            prev_h = next_h

        return torch.stack(outputs), next_h
