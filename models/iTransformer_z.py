import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks.iTransformerEncDec import EncoderLayer
from models.encoders.TransformerEnc import Encoder
from models.blocks.SelfAttention_Family import FullAttention, AttentionLayer
from models.blocks.Embed import DataEmbedding_inverted
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_len):
        super(Model, self).__init__()
        self.seq_len = input_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        d_model = configs.d_model
        self.e_layers = configs.e_layers
        
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(self.seq_len, d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), d_model, configs.n_heads),
                    d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, configs.pred_len, bias=True)
        self.args = configs
        if hasattr(self.args,'norm_style')==False:
            self.args.norm_style='none'
        
    def store_grad(self):
        return
    
    def forecast(self, x_enc, x_mark_enc, z: torch.tensor=None, z_loc=1):
        """
        z: Tensor to add, shape [B, N, E] or [B, L, E] (must match addition location)
        z_loc: Addition location ('input', layer index as int, or 'projection')
        """
        # Copy of original forecast method with z addition points
        feature=None
        _, _, N = x_enc.shape  # B L N
        if z != None and len(z.shape)==2:
            b=x_enc.shape[0]
            # z=z[:b]
            z=z.unsqueeze(0).repeat(b,1,1)
        if len(z.shape)==3:
            z=z[:x_enc.shape[0]]
        if z_loc == 'input2' :
            feature = x_enc.clone().requires_grad_(True)
            if z is not None:
                    x_enc = feature + z
        if self.use_norm and self.args.norm_style == 'none':
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev
        
        
       
        # Input-level addition
        if z_loc == 'input' :
            feature = x_enc.clone().requires_grad_(True)
            if z is not None:
                    x_enc = feature + z
  
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        if z_loc == 'emb' and z is not None:
            feature = enc_out.clone().requires_grad_(True)
            if z_loc is not None:
                enc_out = enc_out + z
        

        # Encoder
        for idx in range(self.e_layers):
            enc_out, attns = self.encoder.attn_layers[idx](enc_out)
            if z_loc == idx+1:
                feature = enc_out.clone().requires_grad_(True)
                if z is not None:
                    enc_out = feature + z
                else:
                    enc_out = feature+torch.zeros_like(feature)
        if self.encoder.norm is not None:
            enc_out = self.encoder.norm(enc_out)

        # Feature-level addition before projection
        
        # Project to output
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if z_loc == 'output':
            feature= dec_out.clone().requires_grad_(True)
            if z is not None:
                dec_out = dec_out + z


        if self.use_norm and self.args.norm_style == 'none':
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        if z_loc == 'output2':
            feature= dec_out.clone().requires_grad_(True)
            if z is not None:
                dec_out = dec_out + z


        return dec_out,feature

    def forward(self, x_enc, x_mark_enc=None, mask=None, z=None, z_loc=None):
        dec_out,feature = self.forecast(x_enc, x_mark_enc, z, z_loc)
        return {'pred': dec_out[:, -self.pred_len:, :],'feature':feature}
    
