# models/decoder.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from .attention import CoverageAttention


class WAPDecoder(nn.Module):
    """
    Conditional GRU with coverage attention (close to the Theano WAP):
      - GRU(emb + prev) -> attention(ctx, state) -> GRU(ctx) -> readout -> maxout -> logits
    Inputs:
      tgt_in:  [L, B]   (teacher-forcing target with <bos>-shifted; int64)
      emb:     nn.Embedding(vocab, dim_word)
      ctx_2d:  [B, C, H, W]
      mask_2d: [B, H, W]
    Returns:
      logits:  [L, B, vocab_size]
      attn_alphas: [L, B, H, W]
    """

    def __init__(
        self,
        vocab_size: int,
        dim_word: int,
        dim_dec: int,
        ctx_channels: int,
        dim_attention: int,
        use_coverage: bool = True,
        dim_coverage: int = 512,
        kernel_coverage: Tuple[int, int] = (5, 5),
        use_dropout: bool = False,
        maxout_groups: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_word = dim_word
        self.dim_dec = dim_dec
        self.use_dropout = use_dropout
        self.maxout_groups = maxout_groups

        # Embedding for target tokens
        self.emb = nn.Embedding(vocab_size, dim_word)

        # Two GRU stages (pre- and post-attention) like original WAP
        self.gru1 = nn.GRU(input_size=dim_word, hidden_size=dim_dec, batch_first=False)
        self.attn = CoverageAttention(
            ctx_channels=ctx_channels,
            state_dim=dim_dec,
            attn_dim=dim_attention,
            use_coverage=use_coverage,
            coverage_dim=dim_coverage,
            kernel_coverage=kernel_coverage,
        )
        self.gru2 = nn.GRU(input_size=ctx_channels, hidden_size=dim_dec, batch_first=False)

        # Readout projections
        self.readout_gru = nn.Linear(dim_dec, dim_word, bias=True)
        self.readout_prev = nn.Linear(dim_word, dim_word, bias=True)
        self.readout_ctx = nn.Linear(ctx_channels, dim_word, bias=True)

        # Maxout + final projection
        assert dim_word % maxout_groups == 0, "dim_word must be divisible by maxout_groups"
        self.proj = nn.Linear(dim_word // maxout_groups, vocab_size, bias=True)
        self.dropout = nn.Dropout(p=0.2) if use_dropout else nn.Identity()

        # Init decoder init-state MLP from mean context
        self.init_state_mlp = nn.Linear(ctx_channels, dim_dec)

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.readout_gru, self.readout_prev, self.readout_ctx, self.proj, self.init_state_mlp]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def init_state_from_ctx(self, ctx_2d: torch.Tensor, mask_2d: torch.Tensor) -> torch.Tensor:
        # masked mean over H,W -> [B, C]
        B, C, H, W = ctx_2d.shape
        mask = mask_2d.unsqueeze(1)  # [B,1,H,W]
        denom = mask.sum(dim=(2, 3)).clamp_min(1.0)  # [B,1]
        mean_ctx = (ctx_2d * mask).sum(dim=(2, 3)) / denom  # [B, C]
        h0 = torch.tanh(self.init_state_mlp(mean_ctx))       # [B, D]
        return h0

    def forward(
        self,
        ctx_2d: torch.Tensor,          # [B, C, H, W]
        mask_2d: torch.Tensor,         # [B, H, W]
        tgt_in: torch.Tensor,          # [L, B] teacher-forcing input (<bos>-shifted)
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        L, B = tgt_in.shape
        device = ctx_2d.device
        C = ctx_2d.size(1)

        # init decoder state
        h1 = self.init_state_from_ctx(ctx_2d, mask_2d)       # [B, D]
        h1 = h1.unsqueeze(0)                                  # GRU expects [1, B, D]
        h2 = h1.clone()

        # coverage tensor
        alpha_past = torch.zeros(B, 1, ctx_2d.size(2), ctx_2d.size(3), device=device)

        emb = self.emb(tgt_in)                                # [L, B, dim_word]
        logits_out = []
        alphas_out = []

        # time-major loop (keeps it transparent & debuggable)
        for t in range(L):
            et = emb[t:t+1]                                   # [1,B,dim_word]
            y1, h1 = self.gru1(et, h1)                        # [1,B,D], [1,B,D]

            # attention
            ctx_vec, alpha_hw, alpha_past = self.attn(
                context_2d=ctx_2d,
                state=h1.squeeze(0),
                mask_2d=mask_2d,
                alpha_past=alpha_past,
            )                                                 # [B,C], [B,H,W], [B,1,H,W]

            y2, h2 = self.gru2(ctx_vec.unsqueeze(0), h2)      # [1,B,D], [1,B,D]

            # readout (logit_gru + logit_prev + logit_ctx)
            ro = self.readout_gru(y2.squeeze(0)) + self.readout_prev(et.squeeze(0)) + self.readout_ctx(ctx_vec)
            if self.use_dropout:
                ro = self.dropout(ro)

            # maxout over groups
            g = self.maxout(ro)                                # [B, dim_word//groups]
            logit = self.proj(g) / temperature                 # [B, vocab]
            logits_out.append(logit.unsqueeze(0))
            alphas_out.append(alpha_hw.unsqueeze(0))

        logits = torch.cat(logits_out, dim=0)                  # [L,B,V]
        alphas = torch.cat(alphas_out, dim=0)                  # [L,B,H,W]
        return logits, alphas

    def maxout(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim_word] -> reshape -> max over 'groups'
        B, D = x.shape
        G = self.maxout_groups
        x = x.view(B, D // G, G)
        x, _ = x.max(dim=-1)
        return x
