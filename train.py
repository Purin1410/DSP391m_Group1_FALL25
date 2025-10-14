# train.py
from __future__ import annotations
import argparse
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
    Callback,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from sconf import Config

from data.datamodule import CROHMEDatamodule


# -------- optional: log gradient norm ----------
class GradNormCallback(Callback):
    @staticmethod
    def gradient_norm(model) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return (total_norm ** 0.5)

    def on_after_backward(self, trainer, pl_module):
        try:
            pl_module.log("train/grad_norm", self.gradient_norm(pl_module), prog_bar=False)
        except Exception:
            pass


def build_logger(cfg: Config):
    backend = 'wandb'
    name = cfg.wandb.get("name", "WAP")
    save_dir = cfg.wandb.get("save_dir", "logs")

    if backend == "wandb":
        try:
            from pytorch_lightning.loggers import WandbLogger
            return WandbLogger(
                project=cfg.wandb.get("project", "DSP391m_Group1_FALL25"),
                name=name,
                save_dir=save_dir,
                config=dict(cfg),
                log_model=cfg.wandb.get("log_model", False),
            )
        except Exception as e:
            print(f"[logger] Wandb not available ({e}), fallback to TensorBoard.")
            return TensorBoardLogger(save_dir=save_dir, name=name, default_hp_metric=False)
    else:
        # fallback
        return TensorBoardLogger(save_dir=save_dir, name=name, default_hp_metric=False)


def build_callbacks(cfg: Config) -> List[Callback]:
    cbs: List[Callback] = []

    # LR monitor
    if cfg.trainer.get("log_lr", True):
        cbs.append(LearningRateMonitor(logging_interval=cfg.trainer.get("lr_logging_interval", "epoch")))

    # Grad norm (optional)
    if cfg.trainer.get("log_grad_norm", False):
        cbs.append(GradNormCallback())

    # Checkpoint "best"
    monitor_metric = cfg.trainer.checkpoint.get("monitor", "val_ExpRate")
    mode = cfg.trainer.checkpoint.get("mode", "max")
    filename = cfg.trainer.checkpoint.get("filename", "{epoch}-{step}-{" + monitor_metric + ":.4f}")
    save_top_k = cfg.trainer.checkpoint.get("save_top_k", 1)
    dirpath = cfg.trainer.checkpoint.get("dirpath", "checkpoints/best")
    cbs.append(
        ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor_metric,
            mode=mode,
            save_top_k=save_top_k,
            save_last=cfg.trainer.checkpoint.get("save_last", True),
            auto_insert_metric_name=False,
        )
    )

    # Checkpoint "all epochs" (optional)
    if cfg.trainer.get("save_all_epochs", False):
        cbs.append(
            ModelCheckpoint(
                dirpath=cfg.trainer.get("all_epochs_dir", "checkpoints/all_epochs"),
                filename="{epoch}-{step}",
                save_top_k=-1,
                every_n_epochs=1,
            )
        )

    # EarlyStopping (optional)
    if cfg.trainer.get("early_stopping", False):
        cbs.append(
            EarlyStopping(
                monitor=cfg.trainer.early_stopping.get("monitor", monitor_metric),
                mode=cfg.trainer.early_stopping.get("mode", mode),
                patience=cfg.trainer.early_stopping.get("patience", 15),
                min_delta=cfg.trainer.early_stopping.get("min_delta", 0.0),
                strict=False,
            )
        )

    return cbs



def train(cfg: Config):
    # 1) seed
    pl.seed_everything(cfg.get("seed_everything", 1337), workers=True)

    dm = CROHMEDatamodule(cfg)
    dm.setup(stage="fit")
    # Init model
    from lit_model import LitWAP
    def build_model(cfg: Config) -> LitWAP:
        lit = LitWAP(
            config=cfg,
            # optimizer
            learning_rate=cfg.model.get("learning_rate", 3e-4),
            weight_decay=cfg.model.get("weight_decay", 0.0),
            betas=tuple(cfg.model.get("betas", [0.9, 0.999])),
            # loss
            use_focal_loss=cfg.model.get("use_focal_loss", False),
            focal_alpha=cfg.model.get("focal_alpha", 1.0),
            focal_gamma=cfg.model.get("focal_gamma", 2.0),
            # beam
            beam_size=cfg.model.get("beam_size", 5),
            max_len=cfg.model.get("max_len", 200),
            alpha=cfg.model.get("alpha", 0.0),
            early_stopping=cfg.model.get("early_stopping", True),
            temperature=cfg.model.get("temperature", 1.0),
            # logging
            log_train_loss_every_step=cfg.trainer.get("log_train_loss_every_step", False),
        )
        return lit

    # 2) Build model
    if cfg.trainer.get("resume_from_checkpoint"):
        print(f"[Trainer] Resuming from checkpoint: {cfg.trainer.resume_from_checkpoint}")
        lit = LitWAP.load_from_checkpoint(cfg.trainer.resume_from_checkpoint, config=cfg)
    else:
        print("[Trainer] Training from new weights")
        lit = build_model(cfg)

    # 4) Logger & Callbacks
    logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    # 5) Trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.get("accelerator", "gpu"),
        devices=cfg.trainer.get("devices", 1),
        precision=cfg.trainer.get("precision", 32),
        max_epochs=cfg.trainer.get("max_epochs", 100),
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.0),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        deterministic=cfg.trainer.get("deterministic", True),
        check_val_every_n_epoch=cfg.trainer.get("check_val_every_n_epoch", 1),
        val_check_interval=cfg.trainer.get("val_check_interval", None),
        num_sanity_val_steps=cfg.trainer.get("num_sanity_val_steps", 2),
        enable_checkpointing=True,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 50),
    )

    # 6) Fit
    ckpt_path = cfg.trainer.get("resume_from_checkpoint", None)
    trainer.fit(lit, datamodule=dm, ckpt_path=ckpt_path)

    # 7) (Optional) Test
    if cfg.trainer.get("run_test_after_fit", False):
        dm.setup(stage="test")
        trainer.test(lit, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = Config(args.config)
    print(cfg.dumps())
    train(cfg)
