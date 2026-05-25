"""
Pre-generate a pool of DDIM-sampled latents from COCO captions.
Each srun task works independently — no DDP/NCCL needed.

Usage:
  srun --ntasks=24 python generate_ddim_latent_pool.py --output_dir /path/to/pool
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append('src')

import utils_model
from ldm.models.diffusion.ddim import DDIMSampler

SCALE_FACTOR = 0.18215


def load_coco_captions(ann_path, seed=0):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    img_to_caption = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_caption:
            img_to_caption[img_id] = ann['caption']
    items = sorted(img_to_caption.items(), key=lambda x: x[0])
    return [it[1] for it in items]


@torch.no_grad()
def generate_pool(args):
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    captions = load_coco_captions(args.coco_ann, seed=args.seed)

    # Build the list of global indices this rank will generate. Two modes:
    #   1. --index_file: explicit list of indices (used for backfilling holes).
    #   2. Default: contiguous block [index_offset, index_offset + num_latents).
    if args.index_file is not None:
        with open(args.index_file, 'r') as f:
            all_indices = [int(line.strip()) for line in f if line.strip()]
        my_indices = all_indices[rank::world_size]
        my_count = len(my_indices)
        if is_main:
            print(f"Index-file mode: {len(all_indices)} total indices across {world_size} tasks")
            print(f"Rank {rank}: {my_count} latents (first={my_indices[0] if my_indices else None}, last={my_indices[-1] if my_indices else None})")
    else:
        per_rank = args.num_latents // world_size
        extra = args.num_latents % world_size
        my_count = per_rank + (1 if rank < extra else 0)
        my_start = args.index_offset + rank * per_rank + min(rank, extra)
        my_indices = list(range(my_start, my_start + my_count))
        if is_main:
            print(f"Generating {args.num_latents} latents across {world_size} tasks")
            print(f"Rank {rank}: {my_count} latents (indices {my_start}–{my_start + my_count - 1})")

    from omegaconf import OmegaConf
    config = OmegaConf.load(args.ldm_config)
    model = utils_model.load_model_from_config(config, args.ldm_ckpt)
    model = model.to(device)
    model.eval()

    sampler = DDIMSampler(model)
    shape = [4, args.latent_size, args.latent_size]

    if is_main:
        print(f"DDIM: steps={args.ddim_steps}, scale={args.scale}, shape={shape}")

    n_batches = (my_count + args.batch_size - 1) // args.batch_size
    generated = 0

    for batch_idx in tqdm(range(n_batches), desc=f"[rank {rank}]", disable=not is_main):
        bs = min(args.batch_size, my_count - generated)
        batch_global_indices = my_indices[generated:generated + bs]

        batch_caps = [captions[idx % len(captions)] for idx in batch_global_indices]
        # Per-batch seeding: derive a deterministic stream from the first index
        # of the batch (preserves the seed=args.seed+global_start convention of
        # the contiguous-range path so a same-index re-generation is
        # reproducible against that path).
        torch.manual_seed(args.seed + batch_global_indices[0])

        c = model.get_learned_conditioning(batch_caps)
        uc = model.get_learned_conditioning([""] * bs)

        samples_z, _ = sampler.sample(
            S=args.ddim_steps, batch_size=bs, shape=shape,
            conditioning=c, verbose=False, eta=0.0,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
        )

        z_raw = samples_z / SCALE_FACTOR

        for i, idx in enumerate(batch_global_indices):
            out_path = os.path.join(args.output_dir, f"{idx:06d}.pt")
            # Skip files that already exist (resume / re-launch safety; especially
            # important in index-file mode where another launcher may have raced).
            if os.path.exists(out_path) and not os.path.islink(out_path):
                continue
            torch.save(z_raw[i].half().cpu(), out_path)

        generated += bs

    if is_main:
        print(f"Rank {rank} done. Generated {generated} latents.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml")
    parser.add_argument("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--coco_ann", type=str, default="/datasets/COCO/022719/annotations/captions_train2017.json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_latents", type=int, default=100000)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--index_offset", type=int, default=0)
    parser.add_argument("--index_file", type=str, default=None,
                        help="If set, read explicit global indices from this file (one per line) "
                             "and generate only those, sharded across SLURM tasks. Overrides "
                             "--index_offset / --num_latents.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_pool(args)


if __name__ == '__main__':
    main()
