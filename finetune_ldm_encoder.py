# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Distill a videoseal-latent watermarker into the Stable Diffusion VAE encoder.

The trained encoder (encoder + quant_conv) learns to produce latents that match a
fixed, watermarked reference latent (produced by `wm.embedder` + `wm.blender` on the
frozen reference encoder's output). The decoder side is fully frozen, so that simply
encoding+decoding an image with the trained encoder embeds a fixed watermark that is
recoverable by the videoseal detector.

Mirrors the structure of `finetune_ldm_decoder.py` (Stable Signature) so the wiring
(LDM loader, COCO dataloader, optimizer, logging, checkpoint save, val loop) stays
consistent. The objective is, however, an L2 distillation in latent space rather than
a HiDDeN bit loss on pixels.
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion

from videoseal.evals.full import setup_model_from_checkpoint


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single GPU / no distributed
        return False, 0, 1, 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return True, rank, world_size, local_rank


def cleanup_distributed():
    if is_distributed():
        dist.destroy_process_group()


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", type=str, default="/datasets/COCO/022719/train2017",
       help="Path to the training data directory")
    aa("--val_dir", type=str, default="/datasets/COCO/022719/val2017",
       help="Path to the validation data directory")

    group = parser.add_argument_group('Model parameters')
    aa("--ldm_config", type=str,
       default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml",
       help="Path to the configuration file for the LDM model")
    aa("--ldm_ckpt", type=str,
       default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt",
       help="Path to the checkpoint file for the LDM model")
    aa("--wm_ckpt", type=str,
       default="/checkpoint/avseal/tuantran/experiments/latent/260519_sd_vae_vseal_coco/expe/checkpoint600.pth",
       help="Path to the trained videoseal-sd watermarking model checkpoint.")
    aa("--wm_nbits", type=int, default=64,
       help="Number of bits in the videoseal watermark message.")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--lambda_wm", type=float, default=1.0,
       help="Weight of the watermarked-latent MSE loss.")
    aa("--reference_model_weight", type=float, default=0.0,
       help="Weight for the optional clean-latent regularizer (mirrors DCAE distillation).")
    aa("--optimizer", type=str, default="AdamW,lr=5e-4",
       help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=100, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=20, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=10, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=1000,
       help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="output/",
       help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def _decode_to_unit_interval(ldm_ae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    """Decode VQGAN-space latents with the frozen decoder and rescale to [0, 1]."""
    imgs = ldm_ae.decode(z)
    imgs_01 = torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1)
    return imgs_01


def _detector_pred(wm_model: nn.Module, imgs_01: torch.Tensor) -> torch.Tensor:
    """Run the videoseal detector on [0,1] pixels and return logits per bit (b, k)."""
    if imgs_01.shape[-2:] != (wm_model.img_size, wm_model.img_size):
        imgs_01 = F.interpolate(
            imgs_01, size=(wm_model.img_size, wm_model.img_size),
            mode='bilinear', align_corners=False
        )
    pred = wm_model.detector(imgs_01)
    if pred.dim() == 4:
        pred = pred.mean(dim=(-2, -1))
    return pred[:, 1:]  # skip the detection-confidence channel


def _bit_acc(pred_logits: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
    """Per-sample bit accuracy. `pred_logits`: (b, k). `msg`: (b, k) in {0, 1}."""
    pred_bits = (pred_logits > 0).float()
    msg_bits = (msg > 0.5).float()
    diff = ~torch.logical_xor(pred_bits > 0.5, msg_bits > 0.5)
    return torch.sum(diff, dim=-1) / diff.shape[-1]


def main(params):

    # Setup distributed training
    distributed, rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # Set seeds for reproductibility
    torch.manual_seed(params.seed + rank)
    torch.cuda.manual_seed_all(params.seed + rank)
    np.random.seed(params.seed + rank)

    # Print the arguments (only on main process)
    if is_main_process():
        print("__git__:{}".format(utils.get_sha()))
        print("__log__:{}".format(json.dumps(vars(params))))
        print(f">>> Distributed training: {distributed}, world_size: {world_size}, rank: {rank}")

    # Create the directories (only on main process)
    if is_main_process():
        if not os.path.exists(params.output_dir):
            os.makedirs(params.output_dir)
        imgs_dir = os.path.join(params.output_dir, 'imgs')
        params.imgs_dir = imgs_dir
        if not os.path.exists(imgs_dir):
            os.makedirs(imgs_dir, exist_ok=True)
    else:
        params.imgs_dir = os.path.join(params.output_dir, 'imgs')

    # Loads LDM auto-encoder models (frozen reference + frozen decoder)
    if is_main_process():
        print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.eval()
    ldm_ae.to(device)
    for param in ldm_ae.parameters():
        param.requires_grad = False

    # Frozen reference autoencoder for clean-latent computation
    ref_ae: AutoencoderKL = deepcopy(ldm_ae)
    ref_ae.eval()
    ref_ae.to(device)
    for param in ref_ae.parameters():
        param.requires_grad = False

    # Loads the trained videoseal watermarker
    if is_main_process():
        print(f'>>> Building videoseal watermarker from {params.wm_ckpt}...')
    wm_model = setup_model_from_checkpoint(params.wm_ckpt)
    wm_model.eval()
    wm_model.to(device)
    for param in wm_model.parameters():
        param.requires_grad = False

    # Resolution sanity check (warning only — embedder is fully convolutional)
    try:
        wm_img_size = int(wm_model.img_size)
    except Exception:
        wm_img_size = None
    if wm_img_size is not None and params.img_size != wm_img_size:
        # Determine downsample factor f used by the SD VAE (typically 8).
        f = getattr(ldm_ae, 'encoder', None)
        # Fall back to 8 if we cannot introspect.
        try:
            f = int(2 ** (len(ldm_ae.encoder.down) - 1))
        except Exception:
            f = 8
        z_c = getattr(ldm_ae, 'z_channels', None) or getattr(getattr(ldm_ae, 'encoder', object()), 'z_channels', 4)
        if is_main_process():
            print(
                f"WARNING: LDM training img_size={params.img_size} differs from watermarker training "
                f"img_size={wm_img_size}; the embedder/blender will run on latents of shape "
                f"({z_c}, {params.img_size // f}, {params.img_size // f}) instead of the shape the "
                f"watermarker was trained on. Distillation should still work because the embedder is "
                f"fully convolutional, but bit-acc may be degraded. Recommend `--img_size {wm_img_size}` "
                f"to exactly match."
            )

    wm_scaling_w = float(wm_model.blender.scaling_w)
    if is_main_process():
        print(f'>>> Watermarker blender.scaling_w = {wm_scaling_w}')

    # Loads the data
    if is_main_process():
        print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])

    # Use DistributedSampler for DDP
    train_dataset = utils.ImageFolder(params.train_dir, transform=vqgan_transform)
    if params.steps:
        # Limit dataset size for step-based training
        num_train_imgs = params.batch_size * params.steps * world_size
        if num_train_imgs < len(train_dataset):
            indices = np.random.choice(len(train_dataset), num_train_imgs, replace=False)
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=params.batch_size, sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True, collate_fn=None
        )
    else:
        train_loader = utils.get_dataloader(
            params.train_dir, vqgan_transform, params.batch_size,
            num_imgs=params.batch_size * params.steps, shuffle=True, num_workers=4,
            collate_fn=None,
        )

    # Validation dataloader (only on main process for metrics, or all processes with DDP)
    val_loader = utils.get_dataloader(
        params.val_dir, vqgan_transform, params.batch_size * 4,
        num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None,
    )

    for ii_key in range(params.num_keys):

        # Sample a fresh fixed watermark message for this key (only on main, then broadcast)
        if is_main_process():
            print(f'\n>>> Sampling watermark message ({params.wm_nbits} bits)...')
        wm_msg = wm_model.get_random_msg(1).to(device)  # (1, k) in {0, 1}
        if distributed:
            dist.broadcast(wm_msg, src=0)
        wm_msg_str = "".join([str(int(b)) for b in wm_msg.view(-1).tolist()])
        if is_main_process():
            print(f'Watermark message: {wm_msg_str}')

        # Copy the LDM and finetune only the encoder + quant_conv
        ldm_encoder: AutoencoderKL = deepcopy(ldm_ae)
        ldm_encoder.decoder = nn.Identity()
        ldm_encoder.post_quant_conv = nn.Identity()
        ldm_encoder.to(device)
        for param in ldm_encoder.parameters():
            param.requires_grad = True

        # Wrap with DDP if distributed
        if distributed:
            ldm_encoder = DDP(ldm_encoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        optim_params = utils.parse_params(params.optimizer)
        # Get parameters from DDP-wrapped model or regular model
        model_params = ldm_encoder.module.parameters() if distributed else ldm_encoder.parameters()
        optimizer = utils.build_optimizer(model_params=model_params, **optim_params)

        # Training loop
        if is_main_process():
            print(f'>>> Training...')
        train_stats = train(
            train_loader, optimizer, ldm_ae, ref_ae, ldm_encoder, wm_model, wm_msg, params,
            distributed, rank, world_size
        )
        # Run validation only on main process to avoid duplicate logging
        if is_main_process():
            val_stats = val(
                val_loader, ldm_ae, ref_ae, ldm_encoder.module if distributed else ldm_encoder, wm_model, wm_msg, params,
            )
        else:
            val_stats = {}

        if is_main_process():
            log_stats = {
                'wm_msg': wm_msg_str,
                'wm_ckpt': params.wm_ckpt,
                'wm_scaling_w': wm_scaling_w,
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
            }
            # Save checkpoint (only on main process)
            # Get state dict from DDP-wrapped model or regular model
            if distributed:
                encoder_state = ldm_encoder.module.state_dict()
            else:
                encoder_state = ldm_encoder.state_dict()

            save_dict = {
                'ldm_encoder': encoder_state,
                'optimizer': optimizer.state_dict(),
                'params': params,
                'wm_msg': wm_msg.cpu().numpy(),
                'wm_msg_str': wm_msg_str,
                'wm_ckpt': params.wm_ckpt,
                'wm_scaling_w': wm_scaling_w,
            }

            ckpt_path = os.path.join(params.output_dir, f"checkpoint_{ii_key:03d}.pth")
            torch.save(save_dict, ckpt_path)
            # Mirror DCAE convention: also dump the msg as .npy next to the checkpoint.
            msg_path = os.path.join(params.output_dir, "watermark_msg.npy")
            np.save(msg_path, wm_msg.cpu().numpy())
            with (Path(params.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            with (Path(params.output_dir) / "keys.txt").open("a") as f:
                f.write(ckpt_path + "\t" + wm_msg_str + "\n")
            print('\n')

    # Cleanup distributed
    cleanup_distributed()


def train(
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    ldm_ae: AutoencoderKL,
    ref_ae: AutoencoderKL,
    ldm_encoder: nn.Module,
    wm_model: nn.Module,
    wm_msg: torch.Tensor,
    params: argparse.Namespace,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")

    # Get the actual model (unwrap DDP if needed)
    encoder_model = ldm_encoder.module if distributed else ldm_encoder
    encoder_model.encoder.train()
    if not isinstance(encoder_model.quant_conv, nn.Identity):
        encoder_model.quant_conv.train()

    base_lr = optimizer.param_groups[0]["lr"]

    # Set epoch for DistributedSampler
    if distributed and hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(0)

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        # Get device from model
        device = next(ldm_encoder.parameters()).device
        imgs = imgs.to(device, non_blocking=True)
        msg_batch = wm_msg.repeat(imgs.shape[0], 1)

        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)

        # Reference forward (frozen)
        with torch.no_grad():
            ref_imgs_z = ref_ae.encode(imgs).mode()  # b z h/f w/f
            preds_w = wm_model.embedder(ref_imgs_z, msg_batch)
            ref_imgs_z_wm = wm_model.blender(ref_imgs_z, preds_w)

        # Trainable forward - use encode method
        if distributed:
            imgs_z = ldm_encoder.module.encode(imgs).mode()
        else:
            imgs_z = ldm_encoder.encode(imgs).mode()

        # Losses
        loss_wm = F.mse_loss(imgs_z, ref_imgs_z_wm)
        if params.reference_model_weight > 0:
            loss_ref = F.mse_loss(imgs_z, ref_imgs_z)
        else:
            loss_ref = torch.zeros((), device=imgs.device)
        loss = params.lambda_wm * loss_wm + params.reference_model_weight * loss_ref

        # Optim step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Cheap on-the-fly bit-accuracy probe (no-grad through the frozen decoder)
        # Only on main process to avoid duplicate computation
        if is_main_process():
            with torch.no_grad():
                imgs_d_01 = _decode_to_unit_interval(ldm_ae, imgs_z.detach())
                pred = _detector_pred(wm_model, imgs_d_01)
                bit_accs = _bit_acc(pred, msg_batch)
                word_accs = (bit_accs == 1).float()
        else:
            bit_accs = torch.zeros(imgs.shape[0], device=imgs.device)
            word_accs = torch.zeros(imgs.shape[0], device=imgs.device)

        # Gather metrics from all processes if distributed
        if distributed:
            # Average losses across processes
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_wm, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_ref, op=dist.ReduceOp.SUM)
            loss /= world_size
            loss_wm /= world_size
            loss_ref /= world_size

        log_stats = {
            "iteration": ii,
            "loss": loss.item(),
            "loss_wm": loss_wm.item(),
            "loss_ref": loss_ref.item(),
            "latent_mse_to_wm": loss_wm.item(),
            "latent_mse_to_ref": F.mse_loss(imgs_z.detach(), ref_imgs_z).item(),
            "bit_acc_avg": torch.mean(bit_accs).item() if is_main_process() else 0.0,
            "word_acc_avg": torch.mean(word_accs).item() if is_main_process() else 0.0,
            "lr": optimizer.param_groups[0]["lr"],
        }
        for name, val_ in log_stats.items():
            metric_logger.update(**{name: val_})
        if ii % params.log_freq == 0 and is_main_process():
            print(json.dumps(log_stats))

        # Save images during training (only on main process)
        if ii % params.save_img_freq == 0 and is_main_process():
            with torch.no_grad():
                imgs_d_clean = _decode_to_unit_interval(ldm_ae, ref_imgs_z)
                imgs_d_trained = _decode_to_unit_interval(ldm_ae, imgs_z.detach())
            save_image(
                torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1),
                os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'),
                nrow=8,
            )
            save_image(
                imgs_d_clean,
                os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'),
                nrow=8,
            )
            save_image(
                imgs_d_trained,
                os.path.join(params.imgs_dir, f'{ii:03}_train_decoded.png'),
                nrow=8,
            )

    if is_main_process():
        print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def val(
    data_loader: Iterable,
    ldm_ae: AutoencoderKL,
    ref_ae: AutoencoderKL,
    ldm_encoder: AutoencoderKL,
    wm_model: nn.Module,
    wm_msg: torch.Tensor,
    params: argparse.Namespace,
):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_encoder.encoder.eval()
    if not isinstance(ldm_encoder.quant_conv, nn.Identity):
        ldm_encoder.quant_conv.eval()

    # Get device from model
    device = next(ldm_ae.parameters()).device

    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device, non_blocking=True)
        msg_batch = wm_msg.repeat(imgs.shape[0], 1)

        ref_imgs_z = ref_ae.encode(imgs).mode()
        imgs_z = ldm_encoder.encode(imgs).mode()

        # Frozen-decoder reconstructions
        imgs_d_clean = ldm_ae.decode(ref_imgs_z)        # baseline reconstruction
        imgs_d_trained = ldm_ae.decode(imgs_z)          # trained-encoder reconstruction

        # Detector accuracy on the trained-encoder output (in [0,1] pixel space).
        imgs_d_trained_01 = torch.clamp(utils_img.unnormalize_vqgan(imgs_d_trained), 0, 1)
        pred = _detector_pred(wm_model, imgs_d_trained_01)
        bit_accs = _bit_acc(pred, msg_batch)
        word_accs = (bit_accs == 1).float()

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_d_trained, imgs_d_clean).mean().item(),
            "latent_mse_to_ref": F.mse_loss(imgs_z, ref_imgs_z).item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            "word_acc_avg": torch.mean(word_accs).item(),
        }
        for name, val_ in log_stats.items():
            metric_logger.update(**{name: val_})

        if ii % params.save_img_freq == 0:
            save_image(
                torch.clamp(utils_img.unnormalize_vqgan(imgs), 0, 1),
                os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'),
                nrow=8,
            )
            save_image(
                torch.clamp(utils_img.unnormalize_vqgan(imgs_d_clean), 0, 1),
                os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'),
                nrow=8,
            )
            save_image(
                imgs_d_trained_01,
                os.path.join(params.imgs_dir, f'{ii:03}_val_decoded.png'),
                nrow=8,
            )

    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = get_parser()
    params = parser.parse_args()
    main(params)
