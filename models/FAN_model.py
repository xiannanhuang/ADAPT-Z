import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, forecast_model, norm_model):
        super().__init__()
        self.args = args
        self.fm = forecast_model
        self.nm = norm_model

    def forward(self, batch_x):
        if self.nm is not None:
            batch_x = self.nm(batch_x, 'n')
        
        if 'former' in self.args.model:
            forecast = self.fm(batch_x)
        else:
            forecast = self.fm(batch_x)
        if isinstance(forecast, dict):
            forecast = forecast['pred']

        if self.nm is not None:
            forecast = self.nm(forecast, 'd')
        
        return forecast