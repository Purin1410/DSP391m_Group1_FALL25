import zipfile
from typing import List, Dict, Any

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch

from datamodule.datamodule import CROHMEDatamodule
from datamodule.utils import Batch
from datamodule.vocab import VocabInfo

from models.comer import CoMER
from utils.utils import (ExpRateRecorder, Hypothesis, ce_loss, to_tgt_output)


class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
        early_stopping: bool = True,
        temperature: float = 1.0,
        vocab_info: VocabInfo = None,
    ):
        super().__init__()
        mcfg = config["model"]
        self.vocab_info = vocab_info
        # Ignore vocab_info in save_hyperparameters to avoid deep serialization issues
        self.save_hyperparameters(ignore=["vocab_info"])
        

        # model
        self.comer_model = CoMER(config, vocab_info=vocab_info)
        #- -------------------------Optimizer config---------------------------------
        self.optimizer_cfg = mcfg.get("optimizer", {})
        self.optimizer_use = self.optimizer_cfg.get("use", "SGD")
        self.exprate_recorder = ExpRateRecorder(vocab_info)

        self.scheduler_cfg = mcfg.get("scheduler", {})
        self.scheduler_use = self.scheduler_cfg.get("use", "ReduceLROnPlateau")
        self.scheduler_interval = self.scheduler_cfg.get("interval", "epoch")
        self.scheduler_monitor = self.scheduler_cfg.get("monitor", "val_ExpRate")

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        return self.comer_model(img, img_mask, tgt)

    def training_step(self, batch: Batch, _):
        out_hat = self(batch.imgs, batch.mask, batch.tgt)

        loss = ce_loss(out_hat, batch.out, ignore_idx=self.vocab_info.pad_id)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        out_hat = self(batch.imgs, batch.mask, batch.tgt)

        loss = ce_loss(out_hat, batch.out, ignore_idx=self.vocab_info.pad_id)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        hyps = self.approximate_joint_search(batch.imgs, batch.mask)

        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        self.log(
            "val_ExpRate",
            self.exprate_recorder,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
    
    def training_epoch_end(self, *args, **kwargs):
        pass
    def validation_epoch_end(self, *args, **kwargs):
        pass
    def training_step_end(self, *args, **kwargs):
        pass
    def validation_step_end(self, *args, **kwargs):
        pass



    def approximate_joint_search(
        self, img: FloatTensor, mask: LongTensor
    ) -> List[Hypothesis]:
        return self.comer_model.beam_search(img, mask, **self.hparams)

    def configure_optimizers(self):
        name = self.optimizer_use
        if name == "SGD":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.optimizer_cfg.get("SGD", {}).get("lr", 0.08),
                momentum=self.optimizer_cfg.get("SGD", {}).get("momentum", 0.9),
                weight_decay=self.optimizer_cfg.get("SGD", {}).get("weight_decay", 1e-4),
            )
        elif name == "Adam":
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.optimizer_cfg.get("Adam", {}).get("lr", 0.08),
                betas=self.optimizer_cfg.get("Adam", {}).get("betas", (0.9, 0.999)),
            )
        elif name == "AdamW":
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.get("AdamW", {}).get("lr", 0.08),
                betas=self.optimizer_cfg.get("AdamW", {}).get("betas", (0.9, 0.999)),
                weight_decay=self.optimizer_cfg.get("AdamW", {}).get("weight_decay", 1e-4),
            )
        elif name == "Adadelta":
            optimizer = optim.Adadelta(
                self.parameters(),
                lr=self.optimizer_cfg.get("Adadelta", {}).get("lr", 1),
                weight_decay=self.optimizer_cfg.get("Adadelta", {}).get("weight_decay", 1e-4),
                eps=self.optimizer_cfg.get("Adadelta", {}).get("eps", 1e-6),
            )
        else:
            raise ValueError(f"Unknown optimizer: {name}")

        sched_name = self.scheduler_use
        if sched_name == "ReduceLROnPlateau":
            reduce_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("mode", "max"),
                factor=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("factor", 0.25),
                patience=self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("patience", 12),
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": self.scheduler_monitor,
            "interval": self.scheduler_interval,
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
