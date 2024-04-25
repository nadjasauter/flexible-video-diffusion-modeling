# WITHOUT LIGHNING (from flexible)
import os
import sys
import argparse
import wandb
import torch.distributed as dist

from improved_diffusion import dist_util
from improved_diffusion.video_datasets import load_data, default_T_dict, default_image_size_dict
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.train_util import TrainLoop
from improved_diffusion.logger import logger




# WITH LIGHNING  (from depthfm)
import os
import sys
import wandb
import torch
import signal
import argparse
import datetime
from omegaconf import OmegaConf

# @Nadja: Lighning modules loaded
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# ddp stuff
from pytorch_lightning.strategies import DDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

#from depthfm.helpers import load_model_weights
#from depthfm.helpers import count_params, exists
#from depthfm.helpers import instantiate_from_config
#from depthfm.trainer_module import TrainerModuleLatentDepthFM

torch.set_float32_matmul_precision('high')



















# parameters
# def create_argparser() from felxible 
# def parse_args() from depth_fm
# parser = argparse.ArgumentParser
def create_argparser():
    defaults = dict(
        dataset="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        resume_id='',  # set this to a previous run's wandb id to resume training
        num_workers=-1,
        pad_with_random_frames=True,
        max_frames=20,
        T=-1,
        sample_interval=50000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser





# load lighning model
class LitNeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        
    def forward(self, x):

    def training_step(self, batch, batch_idx):

    def train_dataloader(self):

    def val_dataloader(self):

    def validation_step(self, batch, batch_idx):
    
    def validation_epoch_end(self, outputs):
    
    def configure_optimizers(self):
# in my case: UNetVideoModel(nn.Module) in improved_diffusion/unet.py -> unet_lightning.py with class UNetVideoModelLighning(nn.Module):






if __name__ == '__main__':
    # Basic lighning structure
    #model = LitNeuralNet(input_size, hidden_size, num_classes)
    # fast_dev_run=True -> runs single batch through training and validation
    #trainer = Trainer(fast_dev_run=True )
    #trainer.fit(model)

    # Depth FM 
    trainer = Trainer(
        logger=logger, 
        callbacks=callbacks,
        **gpu_kwargs,
        # from config
        **OmegaConf.to_container(cfg.train.trainer_params)
    )
    # ....
    # trainer.fit(module, data, ckpt_path=ckpt_path)


    args = create_argparser().parse_args()
    if args.num_workers == -1:
        # Set the number of workers automatically.
        args.num_workers = max(num_available_cores() - 1, 1)
        print(f"num_workers is not specified. It is automatically set to \"number of cores - 1\" = {args.num_workers}")

    # Set T and image size
    video_length = default_T_dict[args.dataset]
    default_T = video_length
    default_image_size = default_image_size_dict[args.dataset]
    args.T = default_T if args.T == -1 else args.T
    args.image_size = default_image_size

    dist_util.setup_dist()
    resume = bool(args.resume_id)
    init_wandb(config=args, id=args.resume_id if resume else None)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # @Nadja: data laoding needs to be changeed
    print("creating data loader...")
    data = load_data(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        T=args.T,
        num_workers=args.num_workers,
    )

    ''' 
    print("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sample_interval=args.sample_interval,
        pad_with_random_frames=args.pad_with_random_frames,
        max_frames=args.max_frames,
        args=args,
    ).run_loop()
    '''