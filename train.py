import pytorch_lightning as pl
from datamodule import CROHMEDatamodule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback
)
import subprocess
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config

class MoreValidationCallback(pl.Callback):
    def __init__(self, monitor="val_ExpRate"):
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, pl_module):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is not None:
            if metric > 0.55:
                trainer.check_val_every_n_epoch = 1

class RcloneUploadCallback(Callback):
    def __init__(self, local_dir, remote_dir):
        super().__init__()
        self.local_dir = local_dir  # Directory to save local checkpoints
        self.remote_dir = remote_dir  # OneDrive remote directory

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % trainer.check_val_every_n_epoch ==0:
            self._rclone_upload()
    
    def _rclone_upload(self):
        print("Training complete. Final upload to OneDrive...")
        command = f"rclone move --update --ignore-existing --no-traverse --verbose {self.local_dir} {self.remote_dir}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during upload: {e}")
        print("Final upload completed.")

def train(config):
    # Seed
    pl.seed_everything(config.seed_everything, workers=True)
    
    # Data
    data_module = CROHMEDatamodule(config = config)

    # Model
    from lit_comer import LitCoMER
    from utils.callbacks import (GradNormCallback)
    if config.trainer.resume_from_checkpoint is not None:
        model_module = LitCoMER.load_from_checkpoint(config.trainer.resume_from_checkpoint, vocab_info=data_module.vocab.get_info())
    else:
        print("Training from new weights")
        model_module = LitCoMER(
            config = config,
            beam_size = config.model.beam_size,
            max_len = config.model.max_len,
            alpha = config.model.alpha,
            early_stopping = config.model.early_stopping,
            temperature = config.model.temperature,
            vocab_info = data_module.vocab.get_info()
        )

   # Logger
    logger = Logger(config.wandb.name, project=config.wandb.project, config=dict(config), log_model=False)
    if config.wandb.get("wandb_watch", False):
        logger.watch(model_module.comer_model, log=config.wandb.get("wandb_watch_log", "gradients"), log_freq=config.wandb.get("wandb_watch_log_freq", 1000))

   # Callback
    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = ModelCheckpoint(save_top_k    = config.trainer.callbacks[1].init_args.save_top_k, 
                                            monitor     = config.trainer.callbacks[1].init_args.monitor,
                                            mode        = config.trainer.callbacks[1].init_args.mode,
                                            filename    = config.trainer.callbacks[1].init_args.filename,
                                            dirpath     = config.trainer.default_root_dir,
                                            )

    rclone_callback = RcloneUploadCallback(
        local_dir = config.trainer.default_root_dir,
        remote_dir = "purin_gdrive:"
    )

    callback = [lr_callback, checkpoint_callback,rclone_callback]

    callback.append(MoreValidationCallback())
    
    if config.trainer.get("log_grad_norm", False):
        grad_norm_callback = GradNormCallback()
        callback.append(grad_norm_callback)
    
    trainer = pl.Trainer(
        gpus                    = config.trainer.gpus,
        accelerator             = config.trainer.accelerator,
        val_check_interval      = config.trainer.val_check_interval,
        check_val_every_n_epoch = config.trainer.check_val_every_n_epoch,
        max_epochs              = config.trainer.max_epochs,
        logger                  = logger,
        deterministic           = config.trainer.deterministic,
        callbacks               = callback,
        # default_root_dir        = config.trainer.default_root_dir,
        resume_from_checkpoint  = config.trainer.resume_from_checkpoint,
        replace_sampler_ddp=False,
    )
    
    trainer.fit(model_module,data_module)

import torch

def enable_tf32_if_tensor_cores_available(device = "cuda", verbose = None) -> bool:
    """
    Check GPU availability and (if Tensor Cores are present) enable TF32 /
    set_float32_matmul_precision("high").

    Returns:
        True if TF32 is enabled, False otherwise
        (no CUDA / no Tensor Cores / fallback case).

    Notes:
    - Call this function once at the beginning of the program before training/evaluation.
    - TF32 / Tensor Cores trade numerical precision for performance.
      Bit-for-bit reproducibility is not guaranteed when TF32 is enabled.
    """
    if not torch.cuda.is_available():
        if verbose:
            print("CUDA is not available — TF32 not enabled.")
        return False

    if device is None:
        device = torch.cuda.current_device()
    try:
        prop = torch.cuda.get_device_properties(device)
    except Exception as e:
        if verbose:
            print(f"Failed to retrieve device properties: {e}")
        return False

    name = prop.name
    major = prop.major
    minor = prop.minor
    compute_capability = major + minor / 10.0

    has_tensor_cores = major >= 7

    if verbose:
        print(f"Device {device}: {name}, compute capability {major}.{minor} ({compute_capability})")
        print(f"Tensor Cores detected (heuristic): {has_tensor_cores}")

    if not has_tensor_cores:
        if verbose:
            print("No Tensor Cores detected by heuristic — keeping default settings.")
        return False

    # Enable TF32 / set precision (if supported)
    # If PyTorch supports torch.set_float32_matmul_precision (>= ~2.0), prefer using it.
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
            if verbose:
                print('Called torch.set_float32_matmul_precision("high").')
        except Exception as e:
            if verbose:
                print("Failed to call set_float32_matmul_precision:", e)

    # Backend flags (for older / compatible PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        print("Cannot activate torch.backends.cuda.matmul.allow_tf32")
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        print("Cannot activate torch.backends.cudnn.allow_tf32")

    if verbose:
        print("TF32 / allow_tf32 has been enabled (if supported by backend).")
        print("WARNING: TF32 trades numerical precision for performance; bit-for-bit reproducibility is lost.")

    return True


if __name__ == "__main__":
    enabled = enable_tf32_if_tensor_cores_available()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    print("TF32 enabled:", enabled)
    train(config)