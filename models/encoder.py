# models/encoder.py
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU, optionally Dropout (sau ReLU)."""
    def __init__(self, in_ch: int, out_ch: int, kernel: Tuple[int, int], dropout: float = 0.0):
        super().__init__()
        padding = tuple(k // 2 for k in kernel)  # "same" padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class PoolPad2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, pad=(0, 1, 0, 1), mode="constant", value=-1.0)  # (left,right,top,bottom)
        x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=False)
        return x


class VGGEncoder(nn.Module):
    """
    VGG-like CNN encoder as in WAP:
    - dim_ConvBlock: List[int], channels per block
    - layersNum_block: List[int], #conv layers in each block
    - kernel_Convenc: Tuple[int,int]
    - After each block: pad(-1) + max pool 2x2
    Returns:
      features: [B, T, C] where T = H'*W' flattened sequence length
      mask_seq: [B, T]
      context_2d: [B, C, H', W']
      mask_2d: [B, H', W']
    """
    def __init__(
        self,
        input_channels: int,
        dim_ConvBlock: List[int],
        layersNum_block: List[int],
        kernel_Convenc: Tuple[int, int] = (3, 1),
        use_dropout: bool = False,
        dropout_p: float = 0.2,
        block_dropout_indices: Tuple[int, int] = (2, 3),
    ):
        super().__init__()
        assert len(dim_ConvBlock) == len(layersNum_block)
        blocks: List[nn.Module] = []
        in_ch = input_channels

        for bi, (out_ch, n_layers) in enumerate(zip(dim_ConvBlock, layersNum_block)):
            for li in range(n_layers):
                apply_dropout = use_dropout and (bi == 3) and (li in (0, 1, 2))
                blocks.append(
                    ConvBlock(in_ch, out_ch, kernel_Convenc, dropout=dropout_p if apply_dropout else 0.0)
                )
                in_ch = out_ch
            blocks.append(PoolPad2d())

        self.backbone = nn.Sequential(*blocks)
        self.out_channels = in_ch

    @staticmethod
    def _downsample_mask(img_mask: torch.Tensor, n_pools: int):
        m = img_mask
        if m.dtype == torch.bool:
            m = m.float()
        for _ in range(n_pools):
            m = m[:, 0::2, 0::2]
        return m

    def forward(self, img: torch.Tensor, img_mask: torch.Tensor):
        """
        img: [B, C_in, H, W]
        img_mask: [B, H, W] (1 valid, 0 pad)
        """
        n_pools = sum(isinstance(m, PoolPad2d) for m in self.backbone)
        mask_2d = self._downsample_mask(img_mask, n_pools)  # [B, H', W']

        ctx_2d = self.backbone(img)                         # [B, C, H', W']
        feat_seq = ctx_2d.flatten(2).transpose(1, 2)        # [B, T=H'*W', C]
        mask_seq = mask_2d.flatten(1)                       # [B, T]
        return feat_seq, mask_seq, ctx_2d, mask_2d
