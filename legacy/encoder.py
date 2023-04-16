import logging

import torch
from torch import nn

from legacy.utils import make_layers


class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._is_encoder = True

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage' + str(index), make_layers(params))
            setattr(self, 'rnn' + str(index), rnn)

    def forward_by_stage(self, input, subnet, rnn):
        seq_number, batch_size, input_channel, height, width = input.size()

        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        outputs_stage, state_stage = rnn(input, None, seq_len=seq_number, is_ashesh=self._is_encoder) # encoder都沒有state

        return outputs_stage, state_stage

    def forward(self, input):
        hidden_states = []
        logging.debug(input.size()) # [batch, Seq, C, H, W]
        input = input.permute(1, 0, 2, 3, 4) # 5D S*B*C*H*W
        for i in range(1, self.blocks + 1):
            input, state_stage = self.forward_by_stage(input, 
                                                       getattr(self, 'stage' + str(i)),
                                                       getattr(self, 'rnn' + str(i)),
                                                      )
            hidden_states.append(state_stage) # 存了三個不同filter_num的[B, filter_num, H', W']
        return tuple(hidden_states)
