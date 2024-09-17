import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
from datetime import datetime
import argparse
import glob
import os
import time
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities import rank_zero_info
from models.transformers.model import Transformer_PL
import torch

logger = logging.getLogger(__name__)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Validation results *****")
        self._log_results(trainer.callback_metrics)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info("***** Test results *****")
        self._log_results(trainer.callback_metrics)
        self._save_results(trainer.callback_metrics, pl_module.hparams.output_dir)

    def _log_results(self, metrics):
        for key, value in sorted(metrics.items()):
            if key not in ["log", "progress_bar"]:
                rank_zero_info(f"{key} = {value}\n")

    def _save_results(self, metrics, output_dir):
        with open(os.path.join(output_dir, "test_results.txt"), "w") as writer:
            for key, value in sorted(metrics.items()):
                if key not in ["log", "progress_bar"]:
                    writer.write(f"{key} = {value}\n")

def add_generic_args(parser, root_dir) -> None:
    parser.add_argument("--offline", action="store_true", default=False, help="Whether to upload to wandb.")
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--min_epochs", default=5, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--accelerator", default="auto", type=str, help="Accelerator to use (auto, cpu, gpu, etc.)")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices to use")
    parser.add_argument("--max_seq_length", default=2000, type=int)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--fp16", default=True, action="store_true")
    parser.add_argument("--gradient_clip_val", default=1.0, type=float)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--dataset", default="yelp", type=str)
    parser.add_argument("--lr_find", action="store_true")
    
def generic_train(
    model: pl.LightningModule,
    args: argparse.Namespace,
    early_stopping_callback=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs
):
    # init model
    odir = Path(model.hparams.output_dir)
    odir.mkdir(exist_ok=True)
    log_dir = Path(os.path.join(model.hparams.output_dir, 'logs'))
    log_dir.mkdir(exist_ok=True)

    # Tensorboard logger
    pl_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_dir,
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
        default_hp_metric=True
    )

    # add custom checkpoints
    ckpt_path = os.path.join(
        args.output_dir, pl_logger.version, "checkpoints",
    )
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_path, filename="{epoch}-{val_loss:.2f}", monitor="val_loss", mode="min", save_top_k=1, verbose=True
        )
    if logging_callback is None:
        logging_callback = LoggingCallback()

    train_params = {
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "callbacks": [logging_callback, checkpoint_callback] + extra_callbacks,
        "logger": pl_logger,
        "gradient_clip_val": args.gradient_clip_val,
    }

    # Check GPU availability
    if torch.cuda.is_available() and args.gpus > 0:
        train_params["gpus"] = min(args.gpus, torch.cuda.device_count())
        if train_params["gpus"] > 1:
            train_params["distributed_backend"] = "ddp"
        train_params["precision"] = 16 if args.fp16 else 32
    else:
        print("No GPUs available. Using CPU for training.")
        # Remove any GPU-related parameters
        train_params.pop("gpus", None)
        train_params.pop("distributed_backend", None)
        train_params.pop("precision", None)

    train_params.update(extra_train_kwargs)

    trainer = pl.Trainer(**train_params)

    if args.lr_find:
        lr_finder = trainer.tuner.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        logger.info(f"Recommended Learning Rate: {new_lr}")
    
    if args.do_train:
        trainer.fit(model)

    return trainer

def main():
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = Transformer_PL.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    print(args)
    pl.seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Check CUDA availability and set accelerator and devices accordingly
    if torch.cuda.is_available() and args.gpus > 0:
        args.accelerator = "gpu"
        args.devices = min(args.gpus, torch.cuda.device_count())
        logger.info(f"Using GPU acceleration with {args.devices} device(s)")
    else:
        args.accelerator = "cpu"
        args.devices = 1  # Set to 1 for CPU, or you can use os.cpu_count() for multi-core CPUs
        logger.info(f"CUDA is not available. Using CPU with {args.devices} core(s)")

    model = Transformer_PL(args)
    trainer = generic_train(model, args)

    if args.do_predict:
        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpointepoch=*.ckpt"), recursive=True)))
        if checkpoints:
            model = model.load_from_checkpoint(checkpoints[-1])
        return trainer.test(model)

if __name__ == "__main__":
    main()