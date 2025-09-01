import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, forecast_model, norm_model):
        super().__init__()
        self.args = args
        self.fm = forecast_model
        self.nm = norm_model

    def forward(self, batch_x,z=None, z_loc=None):
        
        dec_inp = torch.zeros_like(batch_x[:, -self.args.pred_len:, :])

 
        if self.nm is not None:
            batch_x, dec_inp = self.nm(batch_x, 'forward', dec_inp)
        
        if 'former' in self.args.model:
            forecast = self.fm(batch_x,z=z,z_loc=z_loc)
        else:
            forecast = self.fm(batch_x,z=z,z_loc=z_loc)

        if self.nm is not None:
            forecast['pred'] = self.nm(forecast['pred'], 'inverse')
        
        return forecast