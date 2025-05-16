import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.diffMam.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp

from mamba_ssm import Mamba
class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals
    

class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple
    

class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


    

class EncoderLayer(nn.Module):
    def __init__(self, mamba, mamba_r, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.mamba = mamba
        self.mamba_r = mamba_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = AdaLayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, timestep, mask=None, label_emb=None, **kwargs):
        # 新增：AdaLayerNorm条件归一化
        x_ln = self.norm1(x, timestep, label_emb)
        new_x = self.mamba(x_ln) + self.mamba_r(x_ln.flip(dims=[1])).flip(dims=[1])
        x = x + new_x
        x_normed = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(x_normed.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        out = self.norm2(x + y)
        # 核心：输出后用mask屏蔽padding
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()
        return out


class MamEncoder(nn.Module):
    def __init__(self, mamba_layers, norm_layer=None):
        super().__init__()
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.norm = norm_layer


    def forward(self, x, timestep, padding_masks=None, label_emb=None):
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(
                x,
                timestep=timestep,
                mask=padding_masks,
                label_emb=label_emb,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 dropout=0.1,
                 mlp_hidden_times=4,
                 activation='GELU'):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        assert activation in ['GELU', 'GELU2']
        act = nn.GELU() if activation == 'GELU' else GELU2()
        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        self.seasonal = FourierLayer(d_model=n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, timestep=None, mask=None, label_emb=None):
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season

class MamDecoder(nn.Module):
    def __init__(self,
                 n_channel,
                 n_feat,
                 dropout,
                 mlp_hidden_times,
                 activation,
                 n_embd=1024,
                 n_layer=10):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(*[DecoderLayer(
            n_channel=n_channel,
            n_feat=n_feat,
            n_embd=n_embd,
            dropout=dropout,
            mlp_hidden_times=mlp_hidden_times,
            activation=activation
        ) for _ in range(n_layer)])

    def forward(self, x, t, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block in self.blocks:
            x, residual_mean, residual_trend, residual_season = \
                block(x, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)
        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season



class ddpMamba(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=2,
        n_layer_dec=3,
        n_embd=1024,
        mlp_hidden_times=4,
        block_activate='GELU',
        resid_pdrop=0.1,
        max_len=2048,
        conv_params=None,
        **kwargs
    ):
        super().__init__()
        self.emb = Conv_MLP(n_feat, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(
            n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
            padding_mode='circular', bias=False
        )
        self.combine_m = nn.Conv1d(
            n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
            padding_mode='circular', bias=False
        )

        # Mamba Encoder
        self.encoder = MamEncoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=n_embd,
                        d_state=32,
                        d_conv=2,
                        expand=4,
                    ),
                    Mamba(
                        d_model=n_embd,
                        d_state=32,
                        d_conv=2,
                        expand=4,
                    ),
                    d_model=n_embd,
                    d_ff=2048,
                    dropout=resid_pdrop,
                    activation=block_activate
                ) for _ in range(n_layer_enc)
            ],
            norm_layer=nn.LayerNorm(n_embd)
        )
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        # Mamba Decoder
        self.decoder = MamDecoder(
            n_channel=n_channel,
            n_feat=n_feat,
            dropout=resid_pdrop,
            mlp_hidden_times=mlp_hidden_times,
            activation=block_activate,
            n_embd=n_embd,
            n_layer=n_layer_dec,
        )

    def forward(self, input, t, padding_masks=None, return_res=False):
        emb = self.emb(input)
        inp_enc = self.pos_enc(emb)
        inp_enc = self.encoder(inp_enc, t, padding_masks=padding_masks)
        output, mean, trend, season = self.decoder(inp_enc, t, padding_masks=padding_masks)
        res = self.inverse(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        trend = self.combine_m(mean) + res_m + trend

        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m

        return trend, season_error


if __name__ == '__main__':
    pass
