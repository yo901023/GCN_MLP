import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from model.block.graph_frames_yolo import Graph
from model.block.mlp_gcn import Mlp_gcn
import torch.nn.functional as Fnn


class Model(nn.Module):
    def __init__(self, args, svd_path=None):
        super().__init__()
        self.graph = Graph('coco', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32),
                              requires_grad=False)

        self.channel = args.channel
        self.n_joints = args.n_joints      # 17
        self.frames = args.frames          # 51

        # [A*F] -> C
        self.embedding = nn.LazyLinear(self.channel)

        # ========== (NEW) Temporal Self-Attention ==========
        # 你可以在 args 裡加：t_attn_heads, t_attn_layers, t_attn_dropout
        n_heads = getattr(args, "t_attn_heads", 4)
        n_layers = getattr(args, "t_attn_layers", 2)
        attn_drop = getattr(args, "t_attn_dropout", 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.channel,
            nhead=n_heads,
            dim_feedforward=self.channel * 4,
            dropout=attn_drop,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_attn = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # ===================================================

        # mean/std -> 2C -> C
        self.temporal_proj = nn.Linear(self.channel * 2, self.channel)

        self.mlp_gcn = Mlp_gcn(
            args.layers,
            self.channel,
            args.d_hid,
            args.token_dim,
            self.A,
            length=self.n_joints,
            frames=self.frames
        )

        self.head = nn.Linear(self.channel, args.out_channels)  # out_channels=2

    def forward(self, x):
        B, T, A, F = x.shape                 # [B, 51, 6, 4050]

        # 1) flatten + embed: [B, T, A*F] -> [B, T, C]
        x = x.reshape(B, T, A * F)
        x = self.embedding(x)                # [B, 51, C]

        # ========== (NEW) temporal attention on T ==========
        # 讓 51 個時間步彼此做 self-attention（輸出仍 [B,T,C]）
        x = self.temporal_attn(x)            # [B, 51, C]
        # ==================================================

        # 2) summarize over time: 51 -> 1
        mean = x.mean(dim=1, keepdim=True)   # [B,1,C]
        std  = x.std(dim=1, keepdim=True, unbiased=False)  # [B,1,C]
        x = torch.cat([mean, std], dim=-1)   # [B,1,2C]
        x = self.temporal_proj(x)            # [B,1,C]

        # 3) expand to joints: [B,1,C] -> [B,17,C]
        x = x.expand(-1, self.n_joints, -1)

        # 4) GraphMLP + head
        x = self.mlp_gcn(x)                  # [B,17,C]
        x = self.head(x)                     # [B,17,2]
        return x

