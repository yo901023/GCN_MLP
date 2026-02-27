import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
#from model.block.graph_frames import Graph
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
        self.n_joints = args.n_joints      # 例如 17
        self.frames = args.frames          # 例如 51

        # 把 [A * F] 映射到 channel
        self.embedding = nn.LazyLinear(self.channel)

        # 用「平均 + 標準差」做時間摘要，所以維度會變成 2 * channel
        self.temporal_proj = nn.Linear(self.channel * 2, self.channel)

        # GraphMLP: 這邊 length=關節數，frames 用不到時間序列就當作參數傳進去即可
        self.mlp_gcn = Mlp_gcn(
            args.layers,
            self.channel,
            args.d_hid,
            args.token_dim,
            self.A,
            length=self.n_joints,
            frames=self.frames
        )

        self.head = nn.Linear(self.channel, args.out_channels)  # 例如 out_channels=2

    def forward(self, x):
        B, T, A, F = x.shape        # x: [B, 51, 6, 4050]

        # ---- 1. 展平空間維度，做 embedding ----
        x = x.reshape(B, T, A * F)  # [B, 51, 24300]
        x = self.embedding(x)       # [B, 51, C]

        # ---- 2. 在時間維度做摘要：51 -> 1 ----
        # mean: [B, 1, C]
        mean = x.mean(dim=1, keepdim=True)
        # std:  [B, 1, C]
        std = x.std(dim=1, keepdim=True, unbiased=False)

        # 拼在一起變 [B, 1, 2C]
        x = torch.cat([mean, std], dim=-1)

        # 投影回 [B, 1, C]
        x = self.temporal_proj(x)   # [B, 1, C]

        # ---- 3. 複製成 17 個關節 ----
        # 現在每一個關節看到的是「整段動作的摘要」
        x = x.expand(-1, self.n_joints, -1)   # [B, 17, C]

        # ---- 4. 丟進 GraphMLP + head ----
        x = self.mlp_gcn(x)                   # [B, 17, C]
        x = self.head(x)                      # [B, 17, 2]
        return x
        
#目前最好的        
'''
class Model(nn.Module):
    def __init__(self, args, svd_path=None):
        super().__init__()
        self.graph = Graph('coco', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32),
                              requires_grad=False)

        self.channel = args.channel
        self.n_joints = args.n_joints      # 例如 17
        self.frames = args.frames          # 例如 51

        # 把 [A * F] 映射到 channel
        self.embedding = nn.LazyLinear(self.channel)

        # 用「平均 + 標準差」做時間摘要，所以維度會變成 2 * channel
        self.temporal_proj = nn.Linear(self.channel * 2, self.channel)

        # GraphMLP: 這邊 length=關節數，frames 用不到時間序列就當作參數傳進去即可
        self.mlp_gcn = Mlp_gcn(
            args.layers,
            self.channel,
            args.d_hid,
            args.token_dim,
            self.A,
            length=self.n_joints,
            frames=self.frames
        )

        self.head = nn.Linear(self.channel, args.out_channels)  # 例如 out_channels=2

    def forward(self, x):
        B, T, A, F = x.shape        # x: [B, 51, 6, 4050]

        # ---- 1. 展平空間維度，做 embedding ----
        x = x.reshape(B, T, A * F)  # [B, 51, 24300]
        x = self.embedding(x)       # [B, 51, C]

        # ---- 2. 在時間維度做摘要：51 -> 1 ----
        # mean: [B, 1, C]
        mean = x.mean(dim=1, keepdim=True)
        # std:  [B, 1, C]
        std = x.std(dim=1, keepdim=True, unbiased=False)

        # 拼在一起變 [B, 1, 2C]
        x = torch.cat([mean, std], dim=-1)

        # 投影回 [B, 1, C]
        x = self.temporal_proj(x)   # [B, 1, C]

        # ---- 3. 複製成 17 個關節 ----
        # 現在每一個關節看到的是「整段動作的摘要」
        x = x.expand(-1, self.n_joints, -1)   # [B, 17, C]

        # ---- 4. 丟進 GraphMLP + head ----
        x = self.mlp_gcn(x)                   # [B, 17, C]
        x = self.head(x)                      # [B, 17, 2]
        return x
'''  
