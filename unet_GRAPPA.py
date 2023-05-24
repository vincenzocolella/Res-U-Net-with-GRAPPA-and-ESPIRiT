# -*- coding: utf-8 -*-
"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pathlib
from argparse import ArgumentParser
#from pytorch_lightning.loggers import CSVLogger
import pandas as pd

import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type,RandomMaskFunc
from fastMRI.fastmri.data.transforms import UnetDataTransform
from fastMRI.fastmri.data.transforms import UnetDataTransform_Grappa
from fastMRI.fastmri.data.transforms import UnetDataTransform_Test_Grappa
from fastMRI.fastmri.data.transforms import UnetDataTransform_Validation_Grappa
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastMRI.fastmri.pl_modules.unet_module import UnetModule2


def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform_Grappa(args.challenge,args.accelerations, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform_Grappa(args.challenge,args.accelerations, mask_func=mask) ### loro due 
    test_transform = UnetDataTransform_Test_Grappa(args.challenge, args.accelerations) ### modificati
    
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = UnetModule2(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )
    #print(model)
    
    

    # ------------
    # trainer
    # ------------
    #run = neptune.init_run()
    #logger = CSVLogger(save_dir='logs', name='model_log')
    #trainer = pl.Trainer.from_argparse_args(args, logger=logger)
    trainer = pl.Trainer.from_argparse_args(args)
    if args.mode == "train":
        #run = neptune.init_run()
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module) # proviamo
        
    elif args.mode == "test":
        ########## codice aggiunto da me 
        '''path_config = pathlib.Path("fastmri_dirs.yaml")
        default_root_dir = fetch_dir("log_path", path_config) / "unet" / "unet_demo"
        checkpoint_dir = default_root_dir / "checkpoints"
        print(str(checkpoint_dir))
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        print(str(ckpt_list))
        #if ckpt_list:
        #    args.resume_from_checkpoint = str(ckpt_list[-1])
        file_path = str(ckpt_list[-2])
        print('Loaded checkpoint for testing : '+file_path)
        
        #ckpt = torch.load(file_path)
        #model.load_state_dict(torch.load(file_path)['state_dict'], strict=False)
        
       
        model = model.load_from_checkpoint(ckpt)
        #trainer.validate(model, datamodule=data_module, ckpt_path=file_path) # validation with training metrics'''
        trainer.test(model, datamodule=data_module)
        
        
        
    else:
        raise ValueError(f"unrecognized mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("fastmri_dirs.yaml")
    num_gpus = 1
    backend = "ddp"
    batch_size = 1 if backend == "ddp" else num_gpus

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "unet" / "resnet_grappa_demo"

    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="random",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.08],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        type=int,
        help="Acceleration rates to use for masks",
    )

    # data config with path to fastMRI data and batch size
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path=data_path, batch_size=batch_size, test_path=None)

    # module config
    parser = UnetModule2.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    # trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        accelerator=backend,  # what distributed version to use
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=80,  # max number of epochs
    )

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.default_root_dir / "checkpoints",
            save_top_k=True,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.resume_from_checkpoint is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.resume_from_checkpoint = str(ckpt_list[-1])
            print("Loaded checkpoint" + str(ckpt_list[-1]))

    return args



def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
