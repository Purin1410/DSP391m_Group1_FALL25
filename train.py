import pytorch_lightning as pl
from datamodule import CROHMEDatamodule
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger as Logger
import argparse
from sconf import Config



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
                                            filename    = config.trainer.callbacks[1].init_args.filename)

    # Curriculum module
    callback = [lr_callback, checkpoint_callback]
    
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
        default_root_dir        = config.trainer.default_root_dir,
        resume_from_checkpoint  = config.trainer.resume_from_checkpoint,
        replace_sampler_ddp=False,
    )
    
    trainer.fit(model_module,data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    config = Config(args.config)
    print(config.dumps())
    train(config)