# -*- encoding: utf-8 -*-
"""
@File    :   visual_prompt.py    
@Contact :   15047271937.@163.com
@License :   (C)Copyright 2021-2022, Gzhlaker

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/2/20 10:51 AM   Gzhlaker      1.0         None
"""
import torch
from torch import nn

from core.models.layer_norm import LayerNorm
from core.models.t_aggregate import TAggregate
from core.models.temporal_transformer import TemporalTransformer


class VisualPrompt(nn.Module):
    def __init__(self, sim_head, clip_state_dict, T):
        super(VisualPrompt, self).__init__()
        self.sim_header = sim_head
        self.T = T
        assert sim_head in ["meanP", "LSTM", "Transf", "Conv_1D", "Transf_cls"]

        if self.sim_header == "LSTM" or self.sim_header == "Transf" or self.sim_header == "Transf_cls" or self.sim_header == "Conv_1D":
            embed_dim = clip_state_dict["text_projection"].shape[1]

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        if self.sim_header == "Transf":
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print('layer=6')
        if self.sim_header == "LSTM":
            self.lstm_visual = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.apply(self.init_weights)

        if self.sim_header == "Transf_cls":
            self.transformer = TAggregate(clip_length=self.T, embed_dim=embed_dim, n_layers=6)

        if self.sim_header == 'Conv_1D':
            self.shift = nn.Conv1d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False)
            weight = torch.zeros(embed_dim, 1, 3)
            weight[:embed_dim // 4, 0, 0] = 1.0
            weight[embed_dim // 4:embed_dim // 4 + embed_dim // 2, 0, 1] = 1.0
            weight[-embed_dim // 4:, 0, 2] = 1.0
            self.shift.weight = nn.Parameter(weight)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.sim_header == "meanP":
            pass
        elif self.sim_header == 'Conv_1D':
            x_original = x
            x = x.view(-1, c, t)
            x = self.shift(x.float())
            x = x.permute(0, 2, 1)
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original

        elif self.sim_header == "LSTM":
            x_original = x
            x, _ = self.lstm_visual(x.float())
            self.lstm_visual.flatten_parameters()
            x = torch.cat((x, x_original[:, x.size(1):, ...].contiguous()), dim=1)
            x = x.type(x_original.dtype) + x_original
        elif self.sim_header == "Transf_cls":
            x_original = x
            return self.transformer(x).type(x_original.dtype)

        else:
            raise ValueError('Unknown optimizer: {}'.format(self.sim_header))
        return x.mean(dim=1, keepdim=False)
