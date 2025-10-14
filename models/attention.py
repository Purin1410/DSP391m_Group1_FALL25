# models/attention.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoverageAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention with optional coverage, adapted for 2D CNN features.
    Context: [B, C, H, W]
    State  : [B, D]
    Mask   : [B, H, W] -> 1 for valid, 0 for pad

    If coverage is enabled, we convolve alpha_past (B,1,H,W) with conv_Q (kernel_coverage)
    then project to attn_dim and add into the pre-activation.
    """

    def __init__(
        self,
        ctx_channels: int,
        state_dim: int,
        attn_dim: int,
        use_coverage: bool = True,
        coverage_dim: int = 512,
        kernel_coverage: Tuple[int, int] = (5, 5),
    ):
        super().__init__()
        self.ctx_proj = nn.Conv2d(ctx_channels, attn_dim, kernel_size=1, bias=True)
        self.state_proj = nn.Linear(state_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=True)

        self.use_coverage = use_coverage
        if use_coverage:
            self.coverage_conv = nn.Conv2d(
                in_channels=1,
                out_channels=coverage_dim,
                kernel_size=kernel_coverage,
                padding=tuple(k // 2 for k in kernel_coverage),
                bias=True,
            )
            self.coverage_proj = nn.Linear(coverage_dim, attn_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)
        nn.init.xavier_uniform_(self.state_proj.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.v.bias)
        if self.use_coverage:
            nn.init.kaiming_uniform_(self.coverage_conv.weight, nonlinearity="relu")
            nn.init.zeros_(self.coverage_conv.bias)
            nn.init.xavier_uniform_(self.coverage_proj.weight)

    def forward(
        self,
        context_2d: torch.Tensor,     # [B, C, H, W]
        state: torch.Tensor,          # [B, D]
        mask_2d: Optional[torch.Tensor] = None,  # [B, H, W]
        alpha_past: Optional[torch.Tensor] = None,  # [B, 1, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = context_2d.shape

        # Precompute projections
        ctx_proj = self.ctx_proj(context_2d)                # [B, attn, H, W]
        s_proj = self.state_proj(state).unsqueeze(-1).unsqueeze(-1)  # [B, attn, 1, 1]

        e = ctx_proj + s_proj                                # [B, attn, H, W]

        if self.use_coverage:
            if alpha_past is None:
                alpha_past = context_2d.new_zeros(B, 1, H, W)
            cov_feat = F.relu(self.coverage_conv(alpha_past))             # [B, cov, H, W]
            cov_proj = self.coverage_proj(cov_feat.permute(0, 2, 3, 1))   # [B, H, W, attn]
            cov_proj = cov_proj.permute(0, 3, 1, 2)                       # [B, attn, H, W]
            e = e + cov_proj

        e = torch.tanh(e)                                 # [B, attn, H, W]
        scores = self.v(e).squeeze(1)                     # [B, H, W]

        if mask_2d is not None:
            scores = scores.masked_fill(mask_2d == 0, -1e9)

        alpha = torch.softmax(scores.view(B, -1), dim=-1).view(B, H, W)   # [B,H,W]
        alpha_exp = alpha.unsqueeze(1)                                     # [B,1,H,W]
        # context vector: weighted sum over spatial dims -> [B, C]
        ctx_vec = torch.sum(context_2d * alpha_exp, dim=(2, 3))            # [B,C]

        # update coverage
        alpha_past_next = alpha_exp if alpha_past is None else (alpha_past + alpha_exp)

        return ctx_vec, alpha, alpha_past_next
