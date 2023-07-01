import torch, random
from torch import nn

from src.model_architectures.utils import make_layers

class ForecasterPONI(nn.Module):
    def __init__(self, subnets, rnns, target_len, teach_prob, aux_encoder_params):
        super().__init__()
        assert len(subnets) == len(rnns)

        self.blocks = len(subnets)
        self.target_len = target_len
        self.teacher_forcing_ratio = teach_prob
        self.aux_encoder = make_layers(aux_encoder_params)

        for index, (params, rnn) in enumerate(zip(subnets, rnns)):
            setattr(self, 'rnn' + str(self.blocks - index), rnn)
            setattr(self, 'stage' + str(self.blocks - index), make_layers(params))

        print(f'[{self.__class__.__name__}] TargetLen:{self.target_len} ' \
            f'TeacherForcing:{self.teacher_forcing_ratio}')

    def forward_by_stage(self, input_data, state, subnet, rnn):
        _, state_stage = rnn(input_data.unsqueeze(0), state, seq_len=1)
        output = subnet(state_stage)
        return output, state_stage

    def forward(self, hidden_states, aux_rainmap, aux_data = None):
        """
        The original Forecaster doesn't have the input at "stage 3", but PONI
        has. All auxiliary data can be passed into the ForecasterPONI as input.

        Args:
            hidden_states: The output from `Decoder` containing the states
                of the last frame.
            aux_rainmap: Shape of [B, 3, 1, H, W]. The auxiliary rainmap data 
                is treated as the input at "stage 3".
            aux_data: Shape of [B, C, H, W]. The auxiliary heterogeneous data 
                is treated as the input at "stage 3".
        
        Returns:
            fcst_output: The forecast of shape [Seq, Batch, Height, Width]
        """

        # recurrent network
        hidden_states.reverse()
        fcst_output = []
        run_teacher = self.teacher_factor(self.teacher_forcing_ratio)

        for fcst_num in range(self.target_len):
            for stage_num in range(self.blocks, 0, -1): # GRU_layers
                if stage_num == self.blocks:
                    if not run_teacher and fcst_num != 0:
                        input_data = self.pass_through_aux(
                            fcst_num,
                            input_data, 
                            aux_data,
                            from_realtime = True
                        )
                    else:
                        input_data = self.pass_through_aux(
                            fcst_num,
                            aux_rainmap, 
                            aux_data
                        )
                
                input_data, s_state = self.forward_by_stage(
                    input_data,
                    hidden_states[self.blocks - stage_num], 
                    getattr(self, 'stage' + str(stage_num)), 
                    getattr(self, 'rnn' + str(stage_num))
                )
                hidden_states.append(s_state)
            
            # pop out all old states
            [hidden_states.pop(0) for _ in range(self.blocks)]

            # save output
            fcst_output.append(input_data)
        
        fcst_output = torch.stack(fcst_output) #[3, B, C, H, W]
        assert fcst_output.shape[2] == 1, 'Eventually, there should be only one channel'
        return fcst_output[:, :, 0, :, :]

    def teacher_factor(self, ratio):
        return True if random.random() < ratio else False

    def pass_through_aux(self, fcst_hr, rainmap, aux_data, from_realtime=False):
        """
        Args:
            fcst_hr (int): Forecast hour
            rainmap: Rain data to be fed in PONI with shape [B, 3, 1, H, W] or
                [B, 1, H, W]
            aux_data: Heterogeneous data to be fed in PONI with shape [B, C, 
                H, W]
            from_realtime (bool): if the rainmap is from realtime production
        Returns:
            data with shape of [B, C(128), H//30, W//30]
        """
        if aux_data is None:
            if not from_realtime:
                tmp = rainmap[:, fcst_hr]
            else:
                tmp = rainmap
        else:
            if not from_realtime:
                tmp = torch.cat([rainmap[:, fcst_hr], aux_data], dim=1)
            else:
                tmp = torch.cat([rainmap, aux_data], dim=1)
        
        return self.aux_encoder(tmp)
