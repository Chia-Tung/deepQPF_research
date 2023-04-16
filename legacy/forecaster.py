import torch, random
from torch import nn

# from nowcasting.config import cfg
from legacy.utils import make_layers
from legacy.fcst_type import forecasterType


class Forecaster(nn.Module):
    def __init__(self, subnets, rnns, target_len):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._target_len = target_len
        self._is_ashesh = True

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
        print(f'[{self.__class__.__name__}] TargetLen:{self._target_len}')

    def forward_by_stage(self, input, state, subnet, rnn):
        # the kwarg is_encoder refers to the original Ashesh-type simple GRU
        input, state_stage = rnn(input, state, seq_len=self._target_len, is_ashesh = self._is_ashesh)
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))

        return input

    def forward(self, hidden_states):
        input = self.forward_by_stage(None, hidden_states[-1], getattr(self, 'stage3'), getattr(self, 'rnn3'))
        for i in list(range(1, self.blocks))[::-1]:
            input = self.forward_by_stage(input, hidden_states[i - 1], getattr(self, 'stage' + str(i)),
                                          getattr(self, 'rnn' + str(i)))

        assert input.shape[2] == 1, 'Finally, there should be only one channel'
        return input[:, :, 0, :, :]
    
    
class Forecaster_PONI(nn.Module):
    def __init__(self, subnets, rnns, target_len, teach):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self._target_len = target_len
        self._is_PONI = True
        self.teacher_forcing_ratio = teach

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))
        print(f'[{self.__class__.__name__}] TargetLen:{self._target_len} TeacherForcing:{self.teacher_forcing_ratio}')

    def forward_by_stage(self, input, state, subnet, rnn):
        state_stage = rnn(input, state, seq_len=self._target_len, is_PONI=self._is_PONI)
        output = subnet(state_stage) # [B, C, H, W]
        return output, state_stage

    def forward(self, hidden_states, y_add, y_encoder):
        # y -> conv2d
        y_add = torch.permute(y_add, (1, 0, 2, 3)).unsqueeze(2) # [3, B, 1, H, W]
        tar_len, b, ch, h, w = y_add.size()
        y_add = torch.reshape(y_add, (-1, ch, h, w))
        y_input = y_encoder(y_add)
        y_input = torch.reshape(y_input, 
                                (tar_len, b, y_input.size(1), y_input.size(2), y_input.size(3)),
                               )

        # recurrent network
        hidden_states.reverse()
        y_container = []
        if self.teacher_factor(self.teacher_forcing_ratio): # teacher
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3:
                        output = y_input[i]
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j))
                                                        )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        else: # student
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3 and i == 0:
                        output = y_input[i]
                    elif j == 3:
                        output = y_encoder(output)
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j))
                                                        )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        
        y_output = torch.stack(y_container) #[3, B, C, H, W]
        assert y_output.shape[2] == 1, 'Finally, there should be only one channel'
        return y_output[:, :, 0, :, :]

    def teacher_factor(self, ratio):
        return True if random.random() < ratio else False
    
class Forecaster_addPONI(Forecaster_PONI):
    '''
    The advPONI_hetero_from_poni model needs a brand-new forecaster 
    module because the auxiliary information will go through the 
    PONI in each decoder step.
    '''
    def forward(self, hidden_states, y_add, y_encoder, add_info):        
        add_info = torch.mean(add_info, dim=1) # [B, nc, H, W]
        tar_len = y_add.size(1)
        
        # recurrent network
        hidden_states.reverse()
        y_container = []
        if self.teacher_factor(self.teacher_forcing_ratio): # teacher
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3:
                        output = self.pass_through_aux(y_add[:, i:i+1], add_info, y_encoder)
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j)),
                                                           )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        else: # student
            for i in range(tar_len):
                for j in range(3, 0, -1): # GRU_layers
                    if j == 3 and i == 0:
                        output = self.pass_through_aux(y_add[:, i:i+1], add_info, y_encoder)
                    elif j == 3:
                        output = self.pass_through_aux(output, add_info, y_encoder)
                    output, s_state = self.forward_by_stage(output,
                                                            hidden_states[3 - j], 
                                                            getattr(self, 'stage'+str(j)), 
                                                            getattr(self, 'rnn'+str(j))
                                                        )
                    hidden_states.append(s_state)
                [hidden_states.pop(0) for _ in range(3)] # according to GRU layers
                y_container.append(output)
        
        y_output = torch.stack(y_container) #[3, B, C, H, W]
        assert y_output.shape[2] == 1, 'Finally, there should be only one channel'
        return y_output[:, :, 0, :, :]
    
    def pass_through_aux(self, rainmap, aux, y_e):
        # rainmap: [B, 1, H, W]
        # aux: [B, nc, H, W]
        tmp_add = torch.cat([rainmap, aux], dim=1)
        return y_e(tmp_add)