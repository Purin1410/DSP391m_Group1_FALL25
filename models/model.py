# models/model.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn

from .encoder import VGGEncoder
from .decoder import WAPDecoder


class WAPModel(nn.Module):
    """
    High-level WAP: builds encoder + decoder from config and exposes:
      - forward(img, img_mask, tgt_in) -> logits [L,B,V], attn [L,B,H',W']
      - generate(img, img_mask, ...) -> token ids (greedy / beam via generation_utils)
    NOTE: This module does NOT implement training/optimization logic.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        mcfg = config["model"]

        self.vocab_size = int(mcfg["dim_target"])  # <— dùng cho generation_utils

        # ----- Encoder -----
        self.encoder = VGGEncoder(
            input_channels=mcfg["input_channels"],
            dim_ConvBlock=mcfg["dim_ConvBlock"],
            layersNum_block=mcfg["layersNum_block"],
            kernel_Convenc=tuple(mcfg.get("kernel_Convenc", (3, 1))),
            use_dropout=mcfg.get("use_dropout", False),
            block_dropout_indices=tuple(mcfg.get("block_dropout_indices", (3,))),
            dropout_p=mcfg.get("encoder_dropout_p", 0.2),
        )

        # ----- Decoder -----
        self.decoder = WAPDecoder(
            vocab_size=self.vocab_size,
            dim_word=mcfg["dim_word"],
            dim_dec=mcfg["dim_dec"],
            ctx_channels=self.encoder.out_channels,
            dim_attention=mcfg["dim_attention"],
            use_coverage=mcfg.get("use_coverage", True),
            dim_coverage=mcfg.get("dim_coverage", 512),
            kernel_coverage=tuple(mcfg.get("kernel_coverage", (5, 5))),
            use_dropout=mcfg.get("use_dropout", False),
            maxout_groups=mcfg.get("maxout_groups", 2),
        )

    def forward(
        self,
        img: torch.Tensor,          # [B, Cin, H, W]
        img_mask: torch.Tensor,     # [B, H, W] (1 valid, 0 pad)
        tgt_in: torch.Tensor,       # [L, B] (teacher-forcing, shifted)
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, ctx_2d, mask_2d = self.encoder(img, img_mask)
        logits, attn = self.decoder(ctx_2d, mask_2d, tgt_in, temperature=temperature)
        return logits, attn

    @torch.no_grad()
    def generate(
        self,
        img: torch.Tensor,                      # [B, Cin, H, W]
        img_mask: torch.Tensor,                 # [B, H, W]
        bos_id: int,
        eos_id: int,
        max_len: int = 200,
        beam_size: int = 1,
        temperature: float = 1.0,
        alpha: float = 0.0,                     # length-penalty
        early_stopping: bool = True,
    ) -> List[List[int]]:
        """
        Greedy: chạy trực tiếp trên decoder.
        Beam: dùng adapter gọi utils/generation_utils.DecodeModel.beam_search().
        Trả về List[List[int]] token ids.
        """
        device = img.device
        _, _, ctx_2d, mask_2d = self.encoder(img, img_mask)  # [B,C,H',W'], [B,H',W']
        B = img.size(0)

        if beam_size == 1:
            # -------- Greedy (debug-friendly) --------
            results: List[List[int]] = []
            for b in range(B):
                seq = [bos_id]
                h1 = self.decoder.init_state_from_ctx(ctx_2d[b:b+1], mask_2d[b:b+1]).unsqueeze(0)
                h2 = h1.clone()
                alpha_past = torch.zeros(1, 1, ctx_2d.size(2), ctx_2d.size(3), device=device)
                for _ in range(max_len):
                    y_prev = torch.tensor([[seq[-1]]], device=device, dtype=torch.long)  # [1,1]
                    emb = self.decoder.emb(y_prev)                                       # [1,1,Dw]
                    _, h1 = self.decoder.gru1(emb, h1)                                   # [1,1,D]

                    ctx_vec, _, alpha_past = self.decoder.attn(
                        ctx_2d[b:b+1], h1.squeeze(0), mask_2d[b:b+1], alpha_past
                    )
                    y2, h2 = self.decoder.gru2(ctx_vec.unsqueeze(0).unsqueeze(0), h2)    # [1,1,D]
                    ro = (self.decoder.readout_gru(y2.squeeze(0).squeeze(0))
                          + self.decoder.readout_prev(emb.squeeze(0).squeeze(0))
                          + self.decoder.readout_ctx(ctx_vec))
                    if self.decoder.use_dropout:
                        ro = self.decoder.dropout(ro)
                    g = self.decoder.maxout(ro.unsqueeze(0)).squeeze(0)                  # [Dw//g]
                    logit = self.decoder.proj(g) / temperature
                    next_id = int(torch.argmax(logit, dim=-1).item())
                    seq.append(next_id)
                    if next_id == eos_id:
                        break
                if seq[-1] != eos_id:
                    seq.append(eos_id)
                results.append(seq)
            return results

        # -------- Beam search --------
        from data import vocab as _v
        assert bos_id == _v.SOS_IDX and eos_id == _v.EOS_IDX

        from utils.generation_utils import DecodeModel

        class _Adapter(DecodeModel):
            def __init__(self, wap: WAPModel, vocab_size: int):
                super().__init__()
                self.wap = wap.eval()
                self.vocab_size = int(vocab_size)
                self.ctx_2d = None
                self.mask_2d = None

            @torch.no_grad()
            def prepare(self, img: torch.Tensor, img_mask: torch.Tensor):
                _, _, ctx2d, m2d = self.wap.encoder(img, img_mask)
                self.ctx_2d, self.mask_2d = ctx2d, m2d

            @torch.no_grad()
            def transform(
                self,
                src: List[torch.FloatTensor],      # unused 
                src_mask: List[torch.LongTensor],  # unused
                input_ids: torch.LongTensor        # [b, l]
            ) -> torch.FloatTensor:                 # [b, l, vocab]
                assert self.ctx_2d is not None, "Call prepare(img, img_mask) before beam_search"
                b, l = input_ids.shape
                cur_ctx, cur_mask = self.ctx_2d, self.mask_2d
                if cur_ctx.size(0) != b:
                    rep = b // cur_ctx.size(0)
                    cur_ctx = cur_ctx.repeat_interleave(rep, dim=0)
                    cur_mask = cur_mask.repeat_interleave(rep, dim=0)
                logits, _ = self.wap.decoder(cur_ctx, cur_mask, input_ids.transpose(0, 1))
                return logits.transpose(0, 1)  # -> [b, l, vocab]

        adapter = _Adapter(self, self.vocab_size).to(device)
        adapter.prepare(img, img_mask)

        feats = [torch.empty(0, device=device)]
        masks = [torch.empty(0, device=device)]

        hyps = adapter.beam_search(
            src=feats,
            src_mask=masks,
            beam_size=beam_size,
            max_len=max_len,
            alpha=alpha,
            early_stopping=early_stopping,
            temperature=temperature,
        )
        results: List[List[int]] = []
        for h in hyps:
            if hasattr(h, "tokens"):
                tokens = list(h.tokens)
            elif hasattr(h, "ids"):
                tokens = list(h.ids)
            elif hasattr(h, "sequence"):
                tokens = list(h.sequence)
            else:
                try:
                    tokens = list(h)
                except Exception:
                    raise RuntimeError("Unknown Hypothesis format; adding field tokens/ids/sequence.")
            results.append(tokens)
        return results
