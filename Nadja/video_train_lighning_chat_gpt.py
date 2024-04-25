import os
import sys
import argparse
import torch
import pytorch_lightning as pl
import wandb
from datetime import datetime
from improved_diffusion import dist_util
from improved_diffusion.video_datasets import load_data, default_T_dict, default_image_size_dict
from improved_diffusion.resample import create_named_schedule_sampler
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from improved_diffusion.logger import logger


class DiffusionModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        self.schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, self.diffusion)
        self.data = load_data(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            T=args.T,
            num_workers=args.num_workers,
        )
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps
        self.batch_size = args.batch_size
        self.microbatch = args.microbatch
        self.ema_rate = args.ema_rate
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = args.use_fp16
        self.fp16_scale_growth = args.fp16_scale_growth
        self.sample_interval = args.sample_interval
        self.pad_with_random_frames = args.pad_with_random_frames
        self.max_frames = args.max_frames

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Your training step here
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        return self.data


def init_wandb(config, id):
    if torch.distributed.get_rank() != 0:
        return
    wandb_dir = os.environ.get("MY_WANDB_DIR", "none")
    if wandb_dir == "none":
        wandb_dir = None
    wandb.init(
        entity=os.environ['WANDB_ENTITY'],
        project=os.environ['WANDB_PROJECT'],
        name=os.environ['WANDB_RUN'],
        config=config,
        dir=wandb_dir,
        id=id
    )
    print(f"Wandb run id: {wandb.run.id}")
    num_nodes = 1
    if "SLURM_JOB_NODELIST" in os.environ:
        assert "SLURM_JOB_NUM_NODES" in os.environ
        num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        print(f"Node list: {os.environ['SLURM_JOB_NODELIST']}")
    logger.logkv("num_nodes", num_nodes)
    print(f"Number of nodes: {num_nodes}")


def num_available_cores():
    max_num_worker_suggest = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if max_num_worker_suggest is None:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            max_num_worker_suggest = cpu_count
    return max_num_worker_suggest or 1


def main():
    args = create_argparser().parse_args()
    if args.num_workers == -1:
        args.num_workers = max(num_available_cores() - 1, 1)
        print(f"num_workers is not specified. It is automatically set to \"number of cores - 1\" = {args.num_workers}")

    video_length = default_T_dict[args.dataset]
    default_T = video_length
    default_image_size = default_image_size_dict[args.dataset]
    args.T = default_T if args.T == -1 else args.T
    args.image_size = default_image_size

    dist_util.setup_dist()
    resume = bool(args.resume_id)
    init_wandb(config=args, id=args.resume_id if resume else None)

    model = DiffusionModel(args)

    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=args.max_epochs,
        progress_bar_refresh_rate=1,
        log_every_n_steps=args.log_interval,
        # Add other Trainer arguments here
        logger=wandb
    )

    trainer.fit(model)


def create_argparser():
    defaults = dict(
        dataset="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        resume_id='',
        num_workers=-1,
        pad_with_random_frames=True,
        max_frames=20,
        T=-1,
        sample_interval=50000,
        max_epochs=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
