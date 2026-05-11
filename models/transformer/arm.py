import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn.modules.batchnorm import BatchNorm1d


class MaskBatchNorm2d(nn.Module):
    def __init__(self, num_features: int, impl: str = "legacy"):
        super().__init__()
        self.impl = impl
        self.bn = BatchNorm1d(num_features)
        if impl == "groupnorm":
            self.gn = nn.GroupNorm(1, num_features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            [b, d, h, w]
        mask : Tensor
            [b, 1, h, w]

        Returns
        -------
        Tensor
            [b, d, h, w]
        """
        if self.impl == "groupnorm":
            x = x.masked_fill(mask, 0.0)
            return self.gn(x)

        x = rearrange(x, "b d h w -> b h w d")
        mask = mask.squeeze(1)

        not_mask = ~mask

        if self.impl == "legacy":
            flat_x = x[not_mask, :]
            flat_x = self.bn(flat_x)
            x[not_mask, :] = flat_x
        elif self.impl == "masked_vectorized":
            mask_f = not_mask.unsqueeze(-1).to(x.dtype)
            sum_x = (x * mask_f).sum(dim=(0, 1, 2))
            count = mask_f.sum(dim=(0, 1, 2))
            mean = sum_x / count.clamp(min=1.0)
            
            var_x = (((x - mean) * mask_f) ** 2).sum(dim=(0, 1, 2))
            var = var_x / count.clamp(min=1.0)
            
            weight = self.bn.weight
            bias = self.bn.bias
            eps = self.bn.eps
            
            x_norm = (x - mean) / torch.sqrt(var + eps)
            x_norm = x_norm * weight + bias
            
            if self.training and self.bn.track_running_stats:
                momentum = self.bn.momentum
                with torch.no_grad():
                    self.bn.running_mean = (1 - momentum) * self.bn.running_mean + momentum * mean
                    self.bn.running_var = (1 - momentum) * self.bn.running_var + momentum * var * (count / (count - 1).clamp(min=1.0))
            
            x = torch.where(not_mask.unsqueeze(-1), x_norm, x)

        x = rearrange(x, "b h w d -> b d h w")

        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, nhead: int, dc: int, cross_coverage: bool, self_coverage: bool, norm_impl: str = "legacy"):
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
        self.post_norm = MaskBatchNorm2d(nhead, impl=norm_impl)

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

