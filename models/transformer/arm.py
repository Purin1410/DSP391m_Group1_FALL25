import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x: Tensor
            [b, d, h, w]
        mask: Tensor
            [b, 1, h, w], True means padded/invalid.

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)
        not_mask = ~mask

        flat_x = x[not_mask, :]
        flat_x = self.bn(flat_x)
        x[not_mask, :] = flat_x

        x = rearrange(x, "b h w d -> b d h w")
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool):
        super().__init__()
        assert cross_coverage or self_coverage
        self.nhead = nhead
        self.cross_coverage = cross_coverage
        self.self_coverage = self_coverage

        if cross_coverage and self_coverage:
            in_chs = 2 * nhead
        else:
            in_chs = nhead

        self.conv = nn.Conv2d(in_chs, dc, kernel_size=5, padding=2)
        self.act = nn.ReLU(inplace=True)

        self.proj = nn.Conv2d(dc, nhead, kernel_size=1, bias=False)
        self.post_norm = MaskBatchNorm2d(nhead)

    def forward(
        self, prev_attn: Tensor, key_padding_mask: Tensor, h: int, curr_attn: Tensor
    ) -> Tensor:
        """
        Parameters
        ----------
        prev_attn : Tensor
            [(b * nhead), t, l]
        key_padding_mask : Tensor
            [b, l]
        h : int

        Returns
        -------
        Tensor
            [(b * nhead), t, l]
        """
        t = curr_attn.shape[1]
        mask = repeat(key_padding_mask, "b (h w) -> (b t) () h w", h=h, t=t)

        curr_attn = rearrange(curr_attn, "(b n) t l -> b n t l", n=self.nhead)
        prev_attn = rearrange(prev_attn, "(b n) t l -> b n t l", n=self.nhead)

        attns = []
        if self.cross_coverage:
            attns.append(prev_attn)
        if self.self_coverage:
            attns.append(curr_attn)
        attns = torch.cat(attns, dim=1)

        attns = attns.cumsum(dim=2) - attns
        attns = rearrange(attns, "b n t (h w) -> (b t) n h w", h=h)

        cov = self.conv(attns)
        cov = self.act(cov)

        cov = cov.masked_fill(mask, 0.0)
        cov = self.proj(cov)

        cov = self.post_norm(cov, mask)

        cov = rearrange(cov, "(b t) n h w -> (b n) t (h w)", t=t)
        return cov

    def forward_from_sums(
        self,
        prev_attn_sum: Tensor,
        curr_attn_sum: Tensor,
        key_padding_mask: Tensor,
        h: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """Compute ARM bias from pre-accumulated coverage sums for one step.

        Equivalent to ``full_arm(...)[:, -1:, :]`` but operating on fp32
        running sums rather than the full attention history.

        The sums represent cumulative attention *before* the current token
        (i.e., the sum up to but not including the current position).

        Parameters
        ----------
        prev_attn_sum : Tensor  fp32  [B_active, H, S]
            Cumulative sum of *previous-layer* cross-attention weights for all
            prior tokens (layer i-1 final attention).
        curr_attn_sum : Tensor  fp32  [B_active, H, S]
            Cumulative sum of *current-layer* pre-ARM cross-attention weights
            for all prior tokens.
        key_padding_mask : Tensor  [B_original, S]  (bool, True = padded)
        h : int  spatial height of encoder feature map
        dtype : torch.dtype  output dtype to cast back to (model dtype)

        Returns
        -------
        Tensor  [B_active*H, 1, S]  ARM logit correction for the current step
        """
        # prev_attn_sum / curr_attn_sum : fp32 [B, H, S]
        B_active, H, S = prev_attn_sum.shape

        # Build the coverage channel tensor expected by conv:
        # shape [B, n_channels, 1, S] where n_channels = nhead (prev) or 2*nhead (both)
        attns = []
        if self.cross_coverage:
            attns.append(prev_attn_sum)   # [B, H, S]
        if self.self_coverage:
            attns.append(curr_attn_sum)   # [B, H, S]

        # Stack channels: [B, C, S]  where C = nhead or 2*nhead
        cov_flat = torch.cat(attns, dim=1).float()  # ensure fp32

        # Reshape to image format for Conv2d: [B, C, h, w]
        cov_2d = rearrange(cov_flat, "b c (h w) -> b c h w", h=h)

        # Build the padding mask in image format: [B, 1, h, w]
        # key_padding_mask may be from the original batch; gather for active beams.
        # Caller is responsible for passing the already-gathered mask.
        mask_2d = key_padding_mask.float().view(B_active, 1, h, -1).bool()

        cov_out = self.conv(cov_2d)          # [B, dc, h, w]
        cov_out = self.act(cov_out)
        cov_out = cov_out.masked_fill(mask_2d, 0.0)
        cov_out = self.proj(cov_out)         # [B, H, h, w]
        cov_out = self.post_norm(cov_out, mask_2d)
        # Flatten spatial: [B, H, h*w] = [B, H, S]
        cov_out = cov_out.view(B_active, H, S)
        # Rearrange to [B*H, 1, S]
        cov_out = cov_out.view(B_active * H, 1, S)
        return cov_out.to(dtype=dtype)
