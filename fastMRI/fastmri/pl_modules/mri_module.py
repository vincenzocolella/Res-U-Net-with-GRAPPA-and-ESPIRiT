# -*- coding: utf-8 -*-
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""


import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from fastmri.pl_modules import FastMriDataModule, UnetModule
#from fastMRI.fastmri.pl_modules.unet_module import *

import fastmri
import numpy as np
import pytorch_lightning as pl
import torch
from fastmri import evaluate
from torchmetrics.metric import Metric
from fastmri.losses import SSIMLoss
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import matplotlib.pyplot as plt
import os


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity


# +
class MriModule(pl.LightningModule):
    """
    Abstract super class for deep larning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to fastMRI:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers

    Other methods from LightningModule can be overridden as needed.
    """

    def __init__(self, num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()
        
        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
        self.ssim_loss = SSIMLoss(win_size = 7, k1 = 0.01, k2 = 0.03) #ssim
        #self.unet = UnetModule(
        #    1,1
        # )

    def validation_step_end(self, val_logs):
        #image_dir = find_latest_version('/home/colellav/logs/model_log')
        # check inputs
        for k in (
            "batch_idx",
            "fname",
            "slice_num",
            "max_value",
            "output",
            "target",
            "val_loss",
        ):
            if k not in val_logs.keys():
                raise RuntimeError(
                    f"Expected key {k} in dict returned by validation_step."
                )
        if val_logs["output"].ndim == 2:
            val_logs["output"] = val_logs["output"].unsqueeze(0)
        elif val_logs["output"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")
        if val_logs["target"].ndim == 2:
            val_logs["target"] = val_logs["target"].unsqueeze(0)
        elif val_logs["target"].ndim != 3:
            raise RuntimeError("Unexpected output size from validation_step.")

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    : self.num_log_images
                ]
            )

        # log images to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]
        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_idx_{batch_idx}"
                target = val_logs["target"][i].unsqueeze(0)
                output = val_logs["output"][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                fname = val_logs["fname"]
                slice_num = val_logs["slice_num"]
                
                self.log_image(f"{key}/target", target)
                self.log_image(f"{key}/reconstruction", output)
                self.log_image(f"{key}/error", error)
                #target_img_path = os.path.join(image_dir, f"{fname}_slice{slice_num}_target.png")
                #output_img_path = os.path.join(image_dir, f"{fname}_slice{slice_num}_output.png")
                #error_img_path = os.path.join(image_dir, f"{fname}_slice{slice_num}_error.png")
                #plt.imsave(target_img_path, target, cmap='gray')
                #plt.imsave(output_img_path, output, cmap='gray')
                #plt.imsave(error_img_path, error, cmap='gray')

        # compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs["fname"]):
            slice_num = int(val_logs["slice_num"][i].cpu())
            maxval = val_logs["max_value"][i].cpu().numpy()
            output = val_logs["output"][i].cpu().numpy()
            target = val_logs["target"][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target, output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None, ...], output[None, ...], maxval=maxval)
            ).view(1)
            max_vals[fname] = maxval

        return {
            "val_loss": val_logs["val_loss"],
            "mse_vals": dict(mse_vals),
            "target_norms": dict(target_norms),
            "ssim_vals": dict(ssim_vals),
            "max_vals": max_vals,
        }

        
        self.logger.experiment.add_image(name, image, global_step=self.global_step)

    def validation_epoch_end(self, val_logs):
        # aggregate losses
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        # use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]

        # check to make sure we have all files in all metrics
        assert (
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        # apply means across image volumes
        metrics = {"nmse": 0, "ssim": 0, "psnr": 0}
        local_examples = 0
        for fname in mse_vals.keys():
            local_examples = local_examples + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _, v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _, v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics["psnr"] = (
                metrics["psnr"]
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname], dtype=mse_val.dtype, device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics["ssim"] = metrics["ssim"] + torch.mean(
                torch.cat([v.view(-1) for _, v in ssim_vals[fname].items()])
            )

        # reduce across ddp via sum
        metrics["nmse"] = self.NMSE(metrics["nmse"])
        metrics["ssim"] = self.SSIM(metrics["ssim"])
        metrics["psnr"] = self.PSNR(metrics["psnr"])
        tot_examples = self.TotExamples(torch.tensor(local_examples))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses), dtype=torch.float)
        )

        self.log("validation_loss", val_loss / tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}", value / tot_examples)

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step=self.global_step)       
            
    def test_epoch_end(self, test_logs):
        
        outputs = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in test_logs:
            for i, (fname, slice_num) in enumerate(zip(log["fname"], log["slice"])):
                outputs[fname][int(slice_num.cpu())] = log["output"][i]

        # stack all the slices for each file
        for fname in outputs:
            outputs[fname] = np.stack(
                [out for _, out in sorted(outputs[fname].items())]
            )

        # pull the default_root_dir if we have a trainer, otherwise save to cwd
        if hasattr(self, "trainer"):
            save_path = pathlib.Path(self.trainer.default_root_dir) / "reconstructions"
        else:
            save_path = pathlib.Path.cwd() / "reconstructions"
        self.print(f"Saving reconstructionsoooo to {save_path}")

        fastmri.save_reconstructions(outputs, save_path)
        
        #######################
        
        # Aggregate the output dictionaries from each batch into a single dictionary
        results = {"fname": [], "slice": [], "output": [], "mse": 0.0, "psnr": 0.0, "ssim": 0.0}
        num_slices = 0
        for output in test_logs:
            for key, value in output.items():
                if key == "mse":
                    results[key] += value
                elif key == "psnr":
                    results[key] += value
                elif key == "ssim":
                    results[key] += value
                else:
                    results[key].extend(value)
            if isinstance(output["mse"], (list, tuple)):
                num_slices += len(output["mse"])
            else:
                num_slices += 1

        # Compute the mean MSE, PSNR, and SSIM across the entire test dataset
        mean_mse = results["mse"] / num_slices
        mean_psnr = results["psnr"] / num_slices
        mean_ssim = results["ssim"] / num_slices

        # Log the mean values of the metrics
        self.log("test_mse", mean_mse, on_epoch=True)
        self.log("test_psnr", mean_psnr, on_epoch=True)
        self.log("test_ssim", mean_ssim, on_epoch=True)

        # Return a dictionary containing the mean MSE, PSNR, and SSIM
        return {"mean_mse": mean_mse, "mean_psnr": mean_psnr, "mean_ssim": mean_ssim}


    @staticmethod
    
    
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # logging params
        parser.add_argument(
            "--num_log_images",
            default=100, #cambiato da me
            type=int,
            help="Number of images to log to Tensorboard",
        )

        return parser
    
    

###################################### implementa ##############à
    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(input)

        # compute the SSIM loss
        data_range = torch.max(target) - torch.min(target)
        loss = self.SSIMLoss(output, target, data_range)

        self.log("loss", loss.detach())
        return loss

        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        loss = self.ssim_loss(output, batch.target, data_range=1.0, reduced=True)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        image, _, mean, std, fname, slice_num, _ = batch

        # Forward pass through the model
        output = self.forward(image)

        # Compute the mean and standard deviation of the batch
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        # Convert the output to a NumPy array and apply the inverse normalization
        output = (output * std + mean).cpu().numpy()

        # Compute the mean squared error (MSE) between the output and the ground truth
        mse = np.mean((output - image.cpu().numpy())**2)

        # Compute the peak signal-to-noise ratio (PSNR) between the output and the ground truth
        psnr = -10 * np.log10(mse)

        # Compute the Structural Similarity Index (SSIM) between the output and the ground truth
        ssim_value = ssim(image.cpu().numpy().squeeze(), output.squeeze(), data_range=output.max() - output.min())

        # Return a dictionary containing the filename, slice number, output, MSE, PSNR, and SSIM
        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": output,
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim_value
        }


    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=0.001,
            weight_decay=0.0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, 40, 0.1
        )

        return [optim], [scheduler]
# -


'''import os
import re
def find_latest_version(folder_path):
    max_number = -1
    max_folder = None
    for entry in os.scandir(folder_path):
        
        if entry.is_dir():
            folder_name = os.path.basename(entry.path)
            
            try:
                folder_number = int(folder_name[8:])
                
                if folder_number > max_number:
                    max_number = folder_number
                    max_folder = entry.path
                    
            except ValueError:
                pass  # Folder name doesn't contain a number, ignore it
    return(max_folder)

    
print(find_latest_version('/home/colellav/logs/model_log'))'''

# +
s = 'version_12'

s[8:]
# -






