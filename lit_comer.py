import zipfile
from typing import List, Dict, Any

import pytorch_lightning as pl
import torch.optim as optim
from torch import FloatTensor, LongTensor
import torch

from datamodule.datamodule import CROHMEDatamodule
from datamodule.utils import Batch
vocab = CROHMEDatamodule.shared_vocab

from models.comer import CoMER
from utils.utils import (ExpRateRecorder, Hypothesis, ce_loss, to_bi_tgt_out)


class LitCoMER(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
        beam_size: int = 10,
        max_len: int = 200,
        alpha: float = 1.0,
        early_stopping: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        mcfg = config["model"]
        self.save_hyperparameters()
        

        # model
        self.comer_model = CoMER(config)
        #- -------------------------Optimizer ---------------------------------
        self.optimizer_cfg = mcfg.get("optimizer", {})
        self.optimizer_use = self.optimizer_cfg.get("use", "SGD")
        self.optimizer_params = {
            "SGD": 
                optim.SGD(
                    self.parameters(),
                    lr              = self.optimizer_cfg.get("SGD", {}).get("lr", 0.08),
                    momentum        = self.optimizer_cfg.get("SGD", {}).get("momentum", 0.9),
                    weight_decay    = self.optimizer_cfg.get("SGD", {}).get("weight_decay", 1e-4),
                )
            ,
            "Adam": 
                optim.Adam(
                    self.parameters(),
                    lr              = self.optimizer_cfg.get("Adam", {}).get("lr", 0.08),
                    betas           = self.optimizer_cfg.get("Adam", {}).get("betas", (0.9, 0.999)),
                )
            ,
            "AdamW": 
                optim.AdamW(
                    self.parameters(),
                    lr              = self.optimizer_cfg.get("AdamW", {}).get("lr", 0.08),
                    betas           = self.optimizer_cfg.get("AdamW", {}).get("betas", (0.9, 0.999)),
                    weight_decay    = self.optimizer_cfg.get("AdamW", {}).get("weight_decay", 1e-4),
                )
            ,
            "Adadelta": 
                optim.Adadelta(
                    self.parameters(),
                    lr              = self.optimizer_cfg.get("Adadelta", {}).get("lr", 1),
                    weight_decay    = self.optimizer_cfg.get("Adadelta", {}).get("weight_decay", 1e-4),
                    eps             = self.optimizer_cfg.get("Adadelta", {}).get("eps", 1e-6),
                )
            ,
        }
        
        
        #- -------------------------Scheduler ---------------------------------
        self.scheduler_cfg      = mcfg.get("scheduler", {})
        # init main scheduler
        self.scheduler_use      = self.scheduler_cfg.get("use", "ReduceLROnPlateau")
        self.scheduler_interval = self.scheduler_cfg.get("interval", "epoch")
        self.scheduler_monitor  = self.scheduler_cfg.get("monitor", "val_ExpRate")
        # init scheduler params
        self.scheduler_params = {
            "ReduceLROnPlateau": 
                optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer   = self.optimizer_params[self.optimizer_use],
                    mode        = self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("mode", "max"),
                    factor      = self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("factor", 0.25),
                    patience    = self.scheduler_cfg.get("ReduceLROnPlateau", {}).get("patience", 12),
                )
            ,
        }
        

        

        self.exprate_recorder = ExpRateRecorder()

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
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: Batch, _):
        tgt, out = to_bi_tgt_out(batch.indices, self.device)
        out_hat = self(batch.imgs, batch.mask, tgt)

        loss = ce_loss(out_hat, out)
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
        torch.cuda.empty_cache()
    def validation_epoch_end(self, *args, **kwargs):
        torch.cuda.empty_cache()
    def training_step_end(self, *args, **kwargs):
        torch.cuda.empty_cache()
    def validation_step_end(self, *args, **kwargs):
        torch.cuda.empty_cache()

    def test_step(self, batch: Batch, _):
        hyps = self.approximate_joint_search(batch.imgs, batch.mask)
        self.exprate_recorder([h.seq for h in hyps], batch.indices)
        return batch.img_bases, [vocab.indices2label(h.seq) for h in hyps]
    

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
        optimizer = self.optimizer_params[self.optimizer_use]
            

        reduce_scheduler = self.scheduler_params[self.scheduler_use]

        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": self.scheduler_monitor,
            "interval": self.scheduler_interval,
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
