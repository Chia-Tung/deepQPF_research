import torch
from torch import nn

from src.model_architectures.utils import make_layers


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, target_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._target_len = target_len

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, "rnn" + str(self.blocks - index), rnn)
            setattr(self, "stage" + str(self.blocks - index), make_layers(params))
        print(f"[{self.__class__.__name__}] TargetLen:{self._target_len}")

    def forward_by_stage(self, input_data, state, subnet, rnn):
        input_data, _ = rnn(input_data, state, seq_len=self._target_len)
        seq_number, batch_size, input_channel, height, width = input_data.size()
        input_data = torch.reshape(input_data, (-1, input_channel, height, width))
        output_data = subnet(input_data)
        output_data = torch.reshape(
            output_data,
            (
                seq_number,
                batch_size,
                input_data.size(1),
                input_data.size(2),
                input_data.size(3),
            ),
        )

        return output_data

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: The output from `Decoder` containing the states
                of the last frame.

        Returns:
            input_data: The forecast of shape [Seq, Batch, Height, Width]
        """
        input_data = self.forward_by_stage(
            None, hidden_states[-1], getattr(self, "stage3"), getattr(self, "rnn3")
        )
        for i in range(self.blocks - 1, 0, -1):
            input_data = self.forward_by_stage(
                input_data,
                hidden_states[i - 1],
                getattr(self, "stage" + str(i)),
                getattr(self, "rnn" + str(i)),
            )

        assert input_data.shape[2] == 1, "Eventually, there should be only one channel"
        return input_data[:, :, 0, :, :]
