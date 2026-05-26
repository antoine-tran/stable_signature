"""
Reproduce the validation logic from finetune_ldm_decoder.py on 10 random images.

This mimics exactly what the training script does during validation:
1. Load 10 images from COCO val2017
2. Encode through frozen SD VAE -> z
3. Decode z with finetuned decoder -> imgs_w
4. Run VideoWAM detector on imgs_w -> bit_acc

Compare to the OmniSealBench eval which showed chance bit_acc.

Usage:
  python scripts/reproduce_260521_validation.py \
    --decoder_ckpt /path/to/260521/expe/checkpoint_000.pth \
    --extractor_ckpt /path/to/260519/expe/checkpoint300.pth \
    --msg_path /checkpoint/avseal/datasets/latent_imagenet/0903_vseal_imagenet/watermark_msg.npy \
    --src_dir /datasets/COCO/022719/val2017 \
    --num_images 10
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import center_crop, to_tensor

sys.path.insert(0, "/checkpoint/avseal/tuantran/env_srcs/lseal-dc/stable_signature/src")
sys.path.insert(0, "/checkpoint/avseal/tuantran/env_srcs/vseal/videoseal-sd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder_ckpt", required=True)
    ap.add_argument("--extractor_ckpt", required=True)
    ap.add_argument("--msg_path", required=True)
    ap.add_argument("--src_dir", default="/datasets/COCO/022719/val2017")
    ap.add_argument("--num_images", type=int, default=10)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[reproduce] device: {device}")

    # --- Load models ---------------------------------------------------------
    from videoseal.augmentation.neuralcompression import StableDiffusionVAEv1
    from videoseal.models.video_wam import VideoWam
    from videoseal.models.embedder import build_embedder
    from videoseal.models.extractor import build_extractor
    from omegaconf import OmegaConf
    import utils_model

    # Load SD VAE (frozen)
    ldm_config = "sd/stable-diffusion-v-1-4-original/v1-inference.yaml"
    ldm_ckpt = "sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt"
    config = OmegaConf.load(ldm_config)
    ldm_ae = utils_model.load_model_from_config(config, ldm_ckpt)
    ldm_ae = ldm_ae.to(device).eval()
    ldm_aef = ldm_ae.first_stage_model

    # Load finetuned decoder
    state = torch.load(args.decoder_ckpt, map_location="cpu", weights_only=False)
    if "ldm_decoder" in state:
        state = state["ldm_decoder"]
    msg = ldm_aef.load_state_dict(state, strict=False)
    print(f"[reproduce] loaded decoder: {args.decoder_ckpt}")
    print(f"[reproduce] load_state_dict msg: {msg}")

    # Load VideoWAM extractor (same as training)
    emb_cfg = OmegaConf.load(
        "/checkpoint/avseal/tuantran/env_srcs/vseal/videoseal-sd/configs/embedder.yaml"
    )["unet_bottleneck_sd4"]
    embedder = build_embedder("unet_bottleneck_sd4", emb_cfg, nbits=64, hidden_size_multiplier=1)

    ext_cfg = OmegaConf.load(
        "/checkpoint/avseal/tuantran/env_srcs/vseal/videoseal-sd/configs/extractor.yaml"
    )["convnext_tiny"]
    extractor = build_extractor("convnext_tiny", ext_cfg, args.img_size, nbits=64)

    autoencoder = StableDiffusionVAEv1()
    wm_model = VideoWam(
        embedder, extractor, augmenter=None, attenuation=None,
        scaling_w=1.5, scaling_i=1.0,
        img_size=args.img_size, chunk_size=1, step_size=1,
        blending_method="additive",
        autoencoder=autoencoder,
    ).to(device).eval()

    # Load extractor weights
    ckpt = torch.load(args.extractor_ckpt, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    wm_model.load_state_dict(sd, strict=False)
    print(f"[reproduce] loaded extractor: {args.extractor_ckpt}")

    # Load message
    wm_msg = torch.from_numpy(np.load(args.msg_path)).float().to(device)
    if wm_msg.ndim == 1:
        wm_msg = wm_msg.unsqueeze(0)
    print(f"[reproduce] message shape: {wm_msg.shape}, bits: {''.join(str(int(b)) for b in wm_msg[0].cpu().numpy()[:20])}...")

    # Sample images
    random.seed(args.seed)
    files = [f for f in os.listdir(args.src_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    files = random.sample(files, args.num_images)
    print(f"[reproduce] sampled {len(files)} images")

    bit_accs = []
    with torch.no_grad():
        for fname in files:
            img_pil = Image.open(os.path.join(args.src_dir, fname)).convert("RGB")
            w, h = img_pil.size
            scale = args.img_size / min(w, h)
            img_pil = img_pil.resize((int(w * scale + 0.5), int(h * scale + 0.5)), Image.BICUBIC)
            x = to_tensor(img_pil).unsqueeze(0).to(device)
            x = center_crop(x, [args.img_size, args.img_size]).clamp(0, 1)

            # Encode through frozen VAE
            z = ldm_aef.encode(x).mode()

            # Decode with finetuned decoder (this is what was distilled)
            x_w = ldm_aef.decode(z)  # [-1, 1]

            # Run detector (exactly as in finetune_ldm_decoder.py lines 530-541)
            x_w_01 = (x_w * 0.5 + 0.5).clamp(0, 1)
            if x_w_01.shape[-2:] != (wm_model.img_size, wm_model.img_size):
                x_w_01_resized = F.interpolate(x_w_01, size=(wm_model.img_size, wm_model.img_size), mode="bilinear", align_corners=False)
            else:
                x_w_01_resized = x_w_01

            pred_msg = wm_model.detector(x_w_01_resized)
            if pred_msg.dim() == 4:
                pred_msg = pred_msg.mean(dim=(-2, -1))
            pred_msg = pred_msg[:, 1:]

            msg_batch = wm_msg.repeat(pred_msg.shape[0], 1)
            pred_bits = (pred_msg > 0).float()
            diff = (pred_bits == msg_batch).float()
            bit_acc = diff.mean().item()
            bit_accs.append(bit_acc)

            print(f"  {fname}: bit_acc={bit_acc:.4f}")

    print()
    print(f"[reproduce] mean bit_acc over {len(bit_accs)} images: {sum(bit_accs) / len(bit_accs):.4f}")
    print(f"[reproduce] min={min(bit_accs):.4f} max={max(bit_accs):.4f}")


if __name__ == "__main__":
    main()
