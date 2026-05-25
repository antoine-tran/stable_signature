"""
Generate text-to-image samples with SD v1.4 using original + watermarked decoders.

Produces 3 image sets in one pass:
  - samples_nw/       Original decoder (no watermark)
  - samples_posthoc/  Original decoder + VideoSeal posthoc pixel watermark
  - samples_distilled/ Finetuned (distilled) decoder

Multi-GPU via torchrun:
  torchrun --nproc_per_node=8 generate_sd_txt2img.py --output_dir ... --num_imgs 5000
"""

import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

sys.path.append('src')

import utils_model
from ldm.models.diffusion.ddim import DDIMSampler

def setup_distributed():
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    return rank, world_size, device


def load_coco_captions(ann_path, num_imgs=None, seed=0):
    with open(ann_path, 'r') as f:
        data = json.load(f)
    img_to_caption = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_caption:
            img_to_caption[img_id] = ann['caption']
    items = sorted(img_to_caption.items(), key=lambda x: x[0])
    if num_imgs is not None and num_imgs < len(items):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(items), num_imgs, replace=False)
        indices.sort()
        items = [items[i] for i in indices]
    return [it[0] for it in items], [it[1] for it in items]


@torch.no_grad()
def generate(args):
    rank, world_size, device = setup_distributed()
    is_main = (rank == 0)

    output_dir_nw = os.path.join(args.output_dir, 'samples_nw')
    output_dir_posthoc = os.path.join(args.output_dir, 'samples_posthoc')
    output_dir_distilled = os.path.join(args.output_dir, 'samples_distilled')
    skip_posthoc = getattr(args, 'skip_posthoc', False)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(output_dir_nw, exist_ok=True)
        if not skip_posthoc:
            os.makedirs(output_dir_posthoc, exist_ok=True)
        os.makedirs(output_dir_distilled, exist_ok=True)

    if world_size > 1:
        dist.barrier()
    else:
        pass  # dirs already created

    # Load captions (all ranks load same list, then shard)
    if is_main:
        print(f"Loading captions from {args.coco_ann}...")
    img_ids, captions = load_coco_captions(args.coco_ann, num_imgs=args.num_imgs, seed=args.seed)

    if is_main:
        with open(os.path.join(args.output_dir, 'captions.json'), 'w') as f:
            json.dump([{"img_id": iid, "caption": cap} for iid, cap in zip(img_ids, captions)], f, indent=2)
        print(f"Total captions: {len(captions)}, sharding across {world_size} GPUs")

    # Shard by rank
    per_rank = len(captions) // world_size
    start = rank * per_rank
    end = start + per_rank if rank < world_size - 1 else len(captions)
    my_ids = img_ids[start:end]
    my_captions = captions[start:end]

    if is_main:
        print(f"Rank {rank}: generating {len(my_captions)} images (indices {start}–{end-1})")

    # Load SD model
    if is_main:
        print(f"Loading SD model on {device}...")
    config = OmegaConf.load(args.ldm_config)
    model = utils_model.load_model_from_config(config, args.ldm_ckpt)
    model = model.to(device)
    model.eval()

    # Save original decoder state and prepare watermarked decoder state
    orig_first_stage_sd = deepcopy(model.first_stage_model.state_dict())

    if is_main:
        print(f"Loading finetuned decoder from {args.wm_ckpt}...")
    wm_ckpt = torch.load(args.wm_ckpt, map_location='cpu', weights_only=False)
    wm_decoder_sd = wm_ckpt['ldm_decoder'] if 'ldm_decoder' in wm_ckpt else wm_ckpt

    # Load VideoSeal model for posthoc watermarking (skip if not needed)
    vs_model = None
    vs_msg = None
    if not skip_posthoc:
        if is_main:
            print(f"Loading VideoSeal model from {args.vs_ckpt}...")
        from videoseal.evals.full import setup_model_from_checkpoint
        vs_model = setup_model_from_checkpoint(args.vs_ckpt)
        vs_model.blender.scaling_w = args.scaling_w
        vs_model.eval()
        vs_model.to(device)

        vs_msg = torch.from_numpy(np.load(args.vs_msg)).float().to(device)
        if vs_msg.dim() == 1:
            vs_msg = vs_msg.unsqueeze(0)
    elif is_main:
        print("Skipping posthoc watermarking (--skip_posthoc)")

    # Create DDIM sampler
    sampler = DDIMSampler(model)
    shape = [4, args.H // 8, args.W // 8]

    # Use a fixed seed per image (based on img_id) for reproducibility across GPU counts
    n_batches = (len(my_captions) + args.batch_size - 1) // args.batch_size

    for batch_idx in tqdm(range(n_batches), desc=f"[rank {rank}]", disable=not is_main):
        s = batch_idx * args.batch_size
        e = min(s + args.batch_size, len(my_captions))
        batch_caps = my_captions[s:e]
        batch_ids = my_ids[s:e]
        bs = len(batch_caps)

        # Deterministic noise per batch
        torch.manual_seed(args.seed + start + s)

        # Text conditioning
        c = model.get_learned_conditioning(batch_caps)
        uc = model.get_learned_conditioning([""] * bs)

        # DDIM sampling → latents
        samples_z, _ = sampler.sample(
            S=args.ddim_steps, batch_size=bs, shape=shape,
            conditioning=c, verbose=False, eta=args.ddim_eta,
            unconditional_guidance_scale=args.scale,
            unconditional_conditioning=uc,
        )

        # --- Decode with original decoder → samples_nw ---
        model.first_stage_model.load_state_dict(orig_first_stage_sd, strict=True)
        imgs_nw = model.decode_first_stage(samples_z)
        imgs_nw = torch.clamp((imgs_nw + 1.0) / 2.0, 0.0, 1.0)  # [-1,1] → [0,1]

        # --- Apply VideoSeal posthoc → samples_posthoc ---
        imgs_posthoc = None
        if not skip_posthoc:
            msg_batch = vs_msg.repeat(bs, 1)
            imgs_posthoc = vs_model.embed(imgs_nw, msg_batch, is_video=False)["imgs_w"]
            imgs_posthoc = torch.clamp(imgs_posthoc, 0.0, 1.0)

        # --- Decode with finetuned decoder → samples_distilled ---
        model.first_stage_model.load_state_dict(wm_decoder_sd, strict=False)
        imgs_dist = model.decode_first_stage(samples_z)
        imgs_dist = torch.clamp((imgs_dist + 1.0) / 2.0, 0.0, 1.0)

        # Restore original decoder for next iteration
        model.first_stage_model.load_state_dict(orig_first_stage_sd, strict=True)

        # Save image sets
        for i in range(bs):
            fname = f"{batch_ids[i]:012d}.png"
            pairs = [
                (imgs_nw[i], output_dir_nw),
                (imgs_dist[i], output_dir_distilled),
            ]
            if imgs_posthoc is not None:
                pairs.append((imgs_posthoc[i], output_dir_posthoc))
            for img_t, out_dir in pairs:
                arr = (img_t.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(out_dir, fname))

    if world_size > 1:
        dist.barrier()
    if is_main:
        n_nw = len(os.listdir(output_dir_nw))
        n_di = len(os.listdir(output_dir_distilled))
        if skip_posthoc:
            print(f"Generation complete: nw={n_nw}, distilled={n_di}")
        else:
            n_ph = len(os.listdir(output_dir_posthoc))
            print(f"Generation complete: nw={n_nw}, posthoc={n_ph}, distilled={n_di}")

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml")
    parser.add_argument("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--wm_ckpt", type=str, required=True, help="Finetuned decoder checkpoint")
    parser.add_argument("--coco_ann", type=str, required=True, help="COCO captions annotation JSON")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vs_ckpt", type=str, default=None, help="VideoSeal checkpoint for posthoc watermarking")
    parser.add_argument("--vs_msg", type=str, default=None, help="VideoSeal message .npy file")
    parser.add_argument("--scaling_w", type=float, default=0.5, help="VideoSeal posthoc scaling")
    parser.add_argument("--skip_posthoc", action="store_true", help="Skip posthoc watermarking (only generate nw + distilled)")
    parser.add_argument("--num_imgs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=7.5, help="CFG scale")
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args)


if __name__ == '__main__':
    main()
