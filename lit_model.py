# models/lit_model.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import FloatTensor, LongTensor

from models.model import WAPModel
from utils.utils import ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out


class _WAPDecodeAdapter(pl.LightningModule):
    def __init__(self, wap: WAPModel, vocab_size: int):
        super().__init__()
        self.wap = wap.eval()
        self.vocab_size = int(vocab_size)
        self.ctx_2d: Optional[torch.Tensor] = None
        self.mask_2d: Optional[torch.Tensor] = None

    @torch.no_grad()
    def prepare(self, img: FloatTensor, img_mask: LongTensor):
        # Lưu sẵn đặc trưng encoder để dùng nhiều bước trong beam search
        _, _, ctx2d, m2d = self.wap.encoder(img, img_mask)
        self.ctx_2d, self.mask_2d = ctx2d, m2d

    @torch.no_grad()
    def transform(
        self,
        src: List[FloatTensor],        # unused
        src_mask: List[LongTensor],    # unused
        input_ids: LongTensor,         # [b, l] (teacher forcing tokens)
    ) -> FloatTensor:                   # [b, l, vocab_size]
        assert self.ctx_2d is not None and self.mask_2d is not None, \
            "Call prepare(img, img_mask) before beam_search"

        b, _ = input_ids.shape
        cur_ctx, cur_mask = self.ctx_2d, self.mask_2d
        if cur_ctx.size(0) != b:
            rep = b // cur_ctx.size(0)
            cur_ctx = cur_ctx.repeat_interleave(rep, dim=0)
            cur_mask = cur_mask.repeat_interleave(rep, dim=0)

        logits, _ = self.wap.decoder(cur_ctx, cur_mask, input_ids.transpose(0, 1))
        return logits.transpose(0, 1)  # [b, l, vocab]


class LitWAP(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        # optimizer
        learning_rate: float = 3e-4,
        weight_decay: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        # loss
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        # beam search
        beam_size: int = 5,
        max_len: int = 200,
        alpha: float = 0.0,
        early_stopping: bool = True,
        temperature: float = 1.0,
        # logging
        log_train_loss_every_step: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core model (encoder + decoder)
        self.model = WAPModel(config)
        self.vocab_size = self.model.vocab_size 

        # Metrics
        self.exprate_recorder = ExpRateRecorder()

        # Loss selector
        self._use_focal = use_focal_loss
        if use_focal_loss:
            from utils.utils import focal_loss as _focal_loss
            self._focal_fn = _focal_loss

    # ---------- Forward (teacher-forcing) ----------
    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, temperature: float = 1.0
    ) -> FloatTensor:
        """
        img      : [B, C_in, H, W]
        img_mask : [B, H, W]
        tgt      : [B, L]
        return   : [B, L, V]
        """
        logits, _ = self.model(img, img_mask, tgt.transpose(0, 1), temperature=temperature)  # [L, 2B, V]
        return logits.transpose(0, 1)  # [B, L, V]

    # ---------- Training ----------
    def training_step(self, batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)  # [B,L],[B,L]

        out_hat = self(batch.imgs, batch.mask, tgt)  # [B,L,V]
        if self._use_focal:
            loss = self._focal_fn(out_hat, out, alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        else:
            loss = ce_loss(out_hat, out)

        self.log(
            "train_loss",
            loss,
            on_step=self.hparams.log_train_loss_every_step,
            on_epoch=True,
            prog_bar=not self.hparams.log_train_loss_every_step,
            sync_dist=True,
        )
        return loss

    # ---------- Validation ----------
    def validation_step(self, batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)
        if self._use_focal:
            loss = self._focal_fn(out_hat, out, alpha=self.hparams.focal_alpha, gamma=self.hparams.focal_gamma)
        else:
            loss = ce_loss(out_hat, out)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # approximate joint beam search (bi-direction + rescoring)
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        # update ExpRate
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log("val_ExpRate", self.exprate_recorder, prog_bar=True, on_step=False, on_epoch=True)

    # ---------- Test ----------
    def test_step(self, batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        try:
            from data import CROHMEDatamodule
            _v = CROHMEDatamodule.vocab
            pred_texts = [_v.indices2label(h.seq) if hasattr(_v, "indices2label") else h.seq for h in hyps]
        except Exception:
            pred_texts = [h.seq for h in hyps]
        return getattr(batch, "img_bases", None), pred_texts

    def test_epoch_end(self, test_outputs) -> None:
        try:
            exprate = self.exprate_recorder.compute()
            self.print(f"Test ExpRate: {exprate}")
        except Exception:
            pass


    # ---------- Beam search wrapper ----------
    @torch.no_grad()
    def approximate_joint_search(self, img: FloatTensor, mask: LongTensor) -> List[Hypothesis]:
        device = img.device

        adapter = _WAPDecodeAdapter(self.model, vocab_size=self.vocab_size).to(device)
        adapter.prepare(img, mask)

        feats = [torch.empty(0, device=device)]
        masks = [torch.empty(0, device=device)]

        return adapter.beam_search(
            src=feats,
            src_mask=masks,
            beam_size=self.hparams.beam_size,
            max_len=self.hparams.max_len,
            alpha=self.hparams.alpha,
            early_stopping=self.hparams.early_stopping,
            temperature=self.hparams.temperature,
        )

    # ---------- Optimizer / Scheduler ----------
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )

        return optimizer
