import torch, os
import torch.nn as nn

from legacy.adversarial_model_PONI import BalAdvPoniModel

class TargetAttentionLean_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.atten = nn.Sequential(
            nn.Sigmoid(),
            nn.Conv2d(1, 16, 5, padding=2),
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

    def forward(self, input, prediction):
        p1 = prediction[:,None]
        p1 = p1 * self.atten(input)
        return p1[:,0]

class BalAdvPoniAttenModel(BalAdvPoniModel):
    '''
    This model is a PONI+Atten which can add features from input channels
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention = TargetAttentionLean_1()
        
    def forward(self, input, label):
        pred = super().forward(input, label) # pred shape: [Seq, batch, H, W]
        # NOTE most recent rain frame
        output = torch.clone(pred)
        output[0] = self.attention(input[:, -1:, -1], pred[0]) # input shape: [batch, Seq, C, H, W]
        return output
    
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.forecaster.parameters()) +
            list(self.aux.parameters()) + 
            list(self.attention.parameters()),
            lr=self.lr
        )
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr)
        return opt_g, opt_d