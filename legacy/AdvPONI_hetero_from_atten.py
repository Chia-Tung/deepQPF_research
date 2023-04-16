import torch, os
import torch.nn as nn
from legacy.AdvPONI_hetero_from_poni import PoniModel_addponi

class TargetAttentionLean(nn.Module):
    def __init__(self, nc_minus):
        super().__init__()

        self.atten = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(nc_minus, 16, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, add_info, prediction):
        add_info = torch.mean(add_info, dim=1)
        p1 = prediction[:,None]
        p1 = p1 * self.atten(add_info)
        return p1[:,0]

class PoniModel_addatten(PoniModel_addponi):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # the last element of args is nc_minus
        for i in range(3):
            setattr(self, 'attention'+str(i+1), TargetAttentionLean(args[-1]))

        # overwirte self.aux, we only need 1 input channel
        self.aux = torch.nn.Sequential(nn.Conv2d(1, 8, 7, 5, 1), # input C=1, only rainmap
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(8, 32, 5, 3, 1),
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                       nn.Conv2d(32, 128, 3, 2, 1), 
                                       nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      )
        
    def forward(self, input, label, addition):
        '''
        input: [B, 6, 2, H, W]; 
        label: [B, 3, H, W]; 
        addition: [B, 6, N_type, H, W]
        '''
        output_from_ecd = self.encoder(input) # output: [3][B, n_out, H', W']
        poni_rain = self.make_poni_input(input, label) # [B, 3, H, W]
        output_from_fct = self.forecaster(list(output_from_ecd), poni_rain, self.aux)# [3, B, H, W]
        
        # make attention on additional data
        output = torch.clone(output_from_fct)
        for i in range(3):
            output[i] = getattr(self, 'attention'+str(i+1))(addition, output_from_fct[i])
        
        return output
        
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(list(self.encoder.parameters()) +
                             list(self.forecaster.parameters()) +
                             list(self.aux.parameters()) + 
                             list(self.attention1.parameters()) +
                             list(self.attention2.parameters()) +
                             list(self.attention3.parameters()),
                             lr=self.lr
                             )
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d