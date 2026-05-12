import zipfile
from typing import List, Dict, Any, Optional

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
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, rel_ids: Optional[LongTensor] = None
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
        return self.comer_model(img, img_mask, tgt, rel_ids=rel_ids)

    def training_step(self, batch: Batch, _):
        out_hat = self(batch.imgs, batch.mask, batch.tgt, rel_ids=batch.rel_ids)

        loss = ce_loss(out_hat, batch.out, ignore_idx=self.vocab_info.pad_id)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        out_hat = self(batch.imgs, batch.mask, batch.tgt, rel_ids=batch.rel_ids)

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
    
    def on_train_epoch_start(self):
        sampler = None

        datamodule = getattr(self.trainer, "datamodule", None)
        if datamodule is not None:
            sampler = getattr(datamodule, "train_batch_sampler", None)

        if sampler is None:
            loaders = getattr(self.trainer, "train_dataloaders", None)
            if loaders is None:
                loaders = getattr(self.trainer, "train_dataloader", None)
            if loaders is not None and not isinstance(loaders, (list, tuple)):
                loaders = [loaders]
            if loaders:
                for loader in loaders:
                    candidate = getattr(loader, "batch_sampler", None)
                    if hasattr(candidate, "set_epoch"):
                        sampler = candidate
                        break

        if not hasattr(sampler, "set_epoch"):
            raise RuntimeError(
                "Could not find BucketedBatchSampler in on_train_epoch_start; "
                "epoch-dependent shuffling would be frozen."
            )

        sampler.set_epoch(int(self.current_epoch))
    def validation_epoch_end(self, *args, **kwargs):
        pass
    def training_step_end(self, *args, **kwargs):
        pass
    def validation_step_end(self, *args, **kwargs):
        pass

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [self.vocab_info.words.indices2label(h.seq) for h in hyps]

    def test_epoch_end(self, test_outputs) -> None:
        exprate = self.exprate_recorder.compute()
        print(f"Validation ExpRate: {exprate}")

        with zipfile.ZipFile("result.zip", "w") as zip_f:
            for img_bases, preds in test_outputs:
                for img_base, pred in zip(img_bases, preds):
                    content = f"%{img_base}\n${pred}$".encode()
                    with zip_f.open(f"{img_base}.txt", "w") as f:
                        f.write(content)



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
