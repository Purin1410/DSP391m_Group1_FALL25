import pytorch_lightning as pl
from datamodule import CROHMEDatamodule
from lit_comer import LitCoMER
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config
from utils.callbacks import (GradNormCallback)
import subprocess



def train(config):
    # Seed
    pl.seed_everything(config.seed_everything, workers=True)
    
    # Data
    data_module = CROHMEDatamodule(config = config)

    # Model
    if config.trainer.resume_from_checkpoint is not None:
        print("Resuming from checkpoint: ", config.trainer.resume_from_checkpoint)
        model_module = LitCoMER.load_from_checkpoint(config.trainer.resume_from_checkpoint)
    else:
        print("Training from new weights")
        model_module = LitCoMER(
            config = config,
        )

   # Logger
    logger = Logger(config.wandb.name, project=config.wandb.project, config=dict(config), log_model=False)
    logger.watch(model_module.comer_model, log="all", log_freq=100)

   # Callback
    lr_callback = LearningRateMonitor(logging_interval=config.trainer.callbacks[0].init_args.logging_interval)

    checkpoint_callback = ModelCheckpoint(save_top_k    = config.trainer.callbacks[1].init_args.save_top_k, 
                                            monitor     = config.trainer.callbacks[1].init_args.monitor,
                                            mode        = config.trainer.callbacks[1].init_args.mode,
                                            filename    = config.trainer.callbacks[1].init_args.filename)

    grad_norm_callback = GradNormCallback()
    

    # Curriculum module
    callback = [lr_callback, checkpoint_callback, grad_norm_callback]
    
    trainer = pl.Trainer(
        gpus                    = config.trainer.gpus,
        accelerator             = config.trainer.accelerator,
        val_check_interval      = config.trainer.val_check_interval,
        check_val_every_n_epoch = config.trainer.check_val_every_n_epoch,
        max_epochs              = config.trainer.max_epochs,
        logger                  = logger,
        deterministic           = config.trainer.deterministic,
        callbacks               = callback,
        default_root_dir        = config.trainer.default_root_dir,
        resume_from_checkpoint  = config.trainer.resume_from_checkpoint,
    )
    try:
        trainer.fit(model_module,data_module)
    except Exception as e:
        print(f"Training crashed due to: {e}")
    finally:
        print("Ensuring final upload to OneDrive before exit...")
        subprocess.run(f"rclone copy {local_dir} {remote_dir} --update --ignore-existing --verbose", shell=True, check=True)
        print("Final upload completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    train(config)