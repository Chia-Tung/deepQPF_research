import torch
from torch import nn

from src.model_architectures.utils import make_layers


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, "stage" + str(index), make_layers(params))
            setattr(self, "rnn" + str(index), rnn)

    def forward_by_stage(self, input_data, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input_data.size()

        input_data = torch.reshape(input_data, (-1, input_channel, height, width))
        input_data = subnet(input_data)
        input_data = torch.reshape(
            input_data,
            (
                seq_number,
                batch_size,
                input_data.size(1),
                input_data.size(2),
                input_data.size(3),
            ),
        )
        # state = None when encoding
        outputs_stage, state_stage = rnn(input_data, None, seq_len=seq_number)

        return outputs_stage, state_stage

    def forward(self, input_data):
        """
        Args:
            input_data: shape = [B, S, C, H, W]

        Returns:
            hidden_states: The memory of the last frame which is going
                to be passed to the decoder. It's shape is of [N][Batch,
                Channel', Height', Width'] and the channel number of
                each element is different depanding on the `num_filters`
                of GRU.
        """

        input_data = input_data.permute(1, 0, 2, 3, 4)
        hidden_states = []
        for i in range(1, self.blocks + 1):
            input_data, state_stage = self.forward_by_stage(
                input_data,
                getattr(self, "stage" + str(i)),
                getattr(self, "rnn" + str(i)),
            )
            hidden_states.append(state_stage)
        return hidden_states
