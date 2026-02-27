# model/model_with_subcarrier_cross_attn.py
import torch
import torch.nn as nn

# =========================================================
# 1) 全子載波版本：Subcarrier-level (ALL F) cross-attn
#    - 在每個 (B, T, f) 上，沿天線 A 做 leave-one-out attention
#    - 形狀不變：[B, T, A, F] -> [B, T, A, F]
#    ⚠️ 計算量極大，只建議用於概念驗證 / 小 batch / 小 F
# =========================================================
class SubcarrierCrossAttn_All(nn.Module):
    def __init__(self, d_model=64, num_heads=4, n_antennas=6):
        super().__init__()
        self.n_antennas = n_antennas
        self.d_model = d_model

        # scalar -> vector(d_model)
        self.embed = nn.Linear(1, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # vector(d_model) -> scalar
        self.out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, A, F]
        return: [B, T, A, F]
        """
        B, T, A, F = x.shape
        assert A == self.n_antennas, f"Expected A={self.n_antennas}, got {A}"

        # [B, T, A, F] -> [B, T, F, A] -> [B*T*F, A, 1]
        x_f = x.permute(0, 1, 3, 2).contiguous().view(B * T * F, A, 1)

        h = self.embed(x_f)  # [BTF, A, d]

        # leave-one-out mask: 禁止 token i 看自己 i
        attn_mask = torch.eye(A, device=x.device, dtype=h.dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float("-inf"))

        out, _ = self.attn(h, h, h, attn_mask=attn_mask)  # [BTF, A, d]
        out = self.norm(out + h)
        out = self.out(out)  # [BTF, A, 1]

        # back: [B, T, A, F]
        out = out.view(B, T, F, A).permute(0, 1, 3, 2).contiguous()
        return out


# =========================================================
# 2) 分段版本：Band-level cross-attn（實務推薦）
#    - 把 F 切成 K 段，每段 band_size 維
#    - 在每個 (B, T, band) 上沿天線 A 做 leave-one-out attention
#    - 形狀不變：[B, T, A, F] -> [B, T, A, F]
# =========================================================
class SubcarrierCrossAttn_Band(nn.Module):
    def __init__(self, band_size=45, d_model=64, num_heads=4, n_antennas=6):
        super().__init__()
        self.band_size = band_size
        self.d_model = d_model
        self.n_antennas = n_antennas

        # band vector -> embedding
        self.embed = nn.Linear(band_size, d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

        # embedding -> band vector
        self.out = nn.Linear(d_model, band_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, A, F]
        return: [B, T, A, F]
        """
        B, T, A, F = x.shape
        assert A == self.n_antennas, f"Expected A={self.n_antennas}, got {A}"
        assert F % self.band_size == 0, f"F={F} must be divisible by band_size={self.band_size}"

        K = F // self.band_size  # number of bands

        # [B, T, A, F] -> [B, T, A, K, band]
        x_band = x.view(B, T, A, K, self.band_size)

        # -> [B, T, K, A, band] -> [B*T*K, A, band]
        x_band = x_band.permute(0, 1, 3, 2, 4).contiguous().view(B * T * K, A, self.band_size)

        h = self.embed(x_band)  # [BTK, A, d]

        attn_mask = torch.eye(A, device=x.device, dtype=h.dtype)
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float("-inf"))

        out, _ = self.attn(h, h, h, attn_mask=attn_mask)  # [BTK, A, d]
        out = self.norm(out + h)
        out = self.out(out)  # [BTK, A, band]

        # back: [B, T, A, F]
        out = out.view(B, T, K, A, self.band_size).permute(0, 1, 3, 2, 4).contiguous().view(B, T, A, F)
        return out


# =========================================================
# 3) 你的主模型：把子載波 cross-attn 當作前處理 block 嵌回原 pipeline
#    - 你可以選：
#      (a) 用全子載波版（概念）：use_all_subcarrier_attn=True
#      (b) 用分段版（實務）：use_all_subcarrier_attn=False
# =========================================================
class Model(nn.Module):
    def __init__(self, args, svd_path=None):
        super().__init__()
        # --- 基礎配置 ---
        from model.block.graph_frames_yolo import Graph
        from model.block.mlp_gcn import Mlp_gcn

        self.graph = Graph('coco', 'spatial', pad=1)
        self.A_matrix = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)

        self.channel = args.channel
        self.n_joints = args.n_joints  # 17
        self.frames = args.frames      # 51
        self.n_antennas = 6            # A
        self.feat_dim = 4050           # F

        # -----------------------------
        # 子載波 cross-attention 設定
        # -----------------------------
        # 你可以在 args 裡面放這些參數；沒有的話就用預設值
        self.use_all_subcarrier_attn = False  # True=概念版(很重)
        self.band_size =45
        self.sc_d_model = 64
        self.sc_num_heads = 4

        self.subcarrier_attn_all = SubcarrierCrossAttn_All(
            d_model=self.sc_d_model,
            num_heads=self.sc_num_heads,
            n_antennas=self.n_antennas
        )
        self.subcarrier_attn_band = SubcarrierCrossAttn_Band(
            band_size=self.band_size,
            d_model=self.sc_d_model,
            num_heads=self.sc_num_heads,
            n_antennas=self.n_antennas
        )

        # --- 後續 GCN 結構（維持你原本） ---
        # 展平後維度為 A * F = 6 * 4050 = 24300
        self.embedding = nn.Linear(self.n_antennas * self.feat_dim, self.channel)

        self.temporal_proj = nn.Linear(self.channel * 2, self.channel)

        self.mlp_gcn = Mlp_gcn(
            args.layers,
            self.channel,
            args.d_hid,
            args.token_dim,
            self.A_matrix,
            length=self.n_joints,
            frames=self.frames
        )

        self.head = nn.Linear(self.channel, args.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 51, 6, 4050]
        # ---------------------------------
        # Step 1: 子載波 cross-attention
        # ---------------------------------
        if self.use_all_subcarrier_attn:
            # ⚠️ 概念版：全子載波（非常重）
            x = self.subcarrier_attn_all(x)
        else:
            # ✅ 實務版：分段頻帶
            x = self.subcarrier_attn_band(x)

        # ---------------------------------
        # Step 2: 展平並映射到 channel
        # ---------------------------------
        B, T, A, F = x.shape
        x = x.view(B, T, A * F)  # [B, 51, 24300]
        x = self.embedding(x)    # [B, 51, C]

        # ---------------------------------
        # Step 3: 時間摘要 (51 -> 1)
        # ---------------------------------
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        x = torch.cat([mean, std], dim=-1)  # [B, 1, 2C]
        x = self.temporal_proj(x)           # [B, 1, C]

        # ---------------------------------
        # Step 4: 空間 GCN 推理
        # ---------------------------------
        x = x.expand(-1, self.n_joints, -1)  # [B, 17, C]
        x = self.mlp_gcn(x)                  # [B, 17, C]
        x = self.head(x)                     # [B, 17, out_channels]
        return x


