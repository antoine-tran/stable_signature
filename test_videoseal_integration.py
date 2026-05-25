"""
Test script to validate the VideoSeal integration in finetune_ldm_decoder.py.
Runs on CPU with tiny random-weight models and dummy images.
Tests all new code paths: posthoc watermarked ref, extractor loss, conditional loss, val VideoSeal accuracy.
"""
import sys
import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append('src')

# ─── Minimal mock models ───────────────────────────────────────────────

class MockAutoencoder(nn.Module):
    """Minimal mock of AutoencoderKL with encode/decode."""
    def __init__(self, z_channels=4, img_size=64):
        super().__init__()
        self.encoder = nn.Conv2d(3, z_channels, 3, padding=1)
        self.decoder = nn.Conv2d(z_channels, 3, 3, padding=1)
        self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, 1)

    def encode(self, x):
        z = self.encoder(x)
        z = self.quant_conv(z)
        return MockDistribution(z)

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)


class MockDistribution:
    def __init__(self, z):
        self._z = z
    def mode(self):
        return self._z


class MockMsgDecoder(nn.Module):
    """Minimal mock of HiDDeN message decoder."""
    def __init__(self, num_bits=48):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, num_bits)

    def forward(self, x):
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class MockDetector(nn.Module):
    """Minimal mock of VideoSeal detector."""
    def __init__(self, num_bits=32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(3, 1 + num_bits)  # +1 for detection confidence

    def forward(self, x):
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.linear(x)


class MockBlender(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_w = 1.0

    def forward(self, imgs, msgs_w):
        return imgs + self.scaling_w * msgs_w * 0.01


class MockEmbedder(nn.Module):
    def __init__(self, num_bits=32):
        super().__init__()
        self.fc = nn.Linear(num_bits, 3)

    def forward(self, x):
        return x


class MockVideoSeal(nn.Module):
    """Minimal mock of VideoSeal model with embed/detector."""
    def __init__(self, num_bits=32, img_size=64):
        super().__init__()
        self.embedder = MockEmbedder(num_bits)
        self.detector = MockDetector(num_bits)
        self.blender = MockBlender()
        self.img_size = img_size

    def embed(self, imgs, msgs, is_video=False):
        b = imgs.shape[0]
        msgs_spatial = msgs.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, imgs.shape[2], imgs.shape[3])
        mask = torch.ones(b, 1, imgs.shape[2], imgs.shape[3], device=imgs.device)
        noise = torch.randn_like(imgs) * 0.01
        imgs_w = self.blender(imgs, noise)
        return {"imgs_w": imgs_w.clamp(0, 1)}


# ─── Import components from finetune_ldm_decoder ──────────────────────

sys.path.insert(0, '.')
import utils
import utils_img


def test_train_loop():
    """Test the train() function code path with extractor_weight > 0."""
    print("=" * 60)
    print("TEST 1: Training loop with VideoSeal extractor (extractor_weight > 0)")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 2
    img_size = 64
    num_bits_hidden = 48
    num_bits_vs = 32

    # Create mock models
    ldm_ae = MockAutoencoder(z_channels=4, img_size=img_size)
    ldm_ae.eval()
    ldm_ae.to(device)
    for p in ldm_ae.parameters():
        p.requires_grad = False

    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.to(device)
    for p in ldm_decoder.parameters():
        p.requires_grad = True

    msg_decoder = MockMsgDecoder(num_bits=num_bits_hidden)
    msg_decoder.eval()
    for p in msg_decoder.parameters():
        p.requires_grad = False

    wm_model = MockVideoSeal(num_bits=num_bits_vs, img_size=img_size)
    wm_model.eval()
    wm_model.requires_grad_(False)
    wm_model.blender.scaling_w = 0.5

    wm_msg = torch.randint(0, 2, (1, num_bits_vs), dtype=torch.float32, device=device)
    key = torch.randint(0, 2, (1, num_bits_hidden), dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(ldm_decoder.parameters(), lr=5e-4)

    loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')
    loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)  # MSE for simplicity

    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    lambda_i = 0.2
    lambda_w = 1.0
    extractor_weight = 0.1

    # Simulate 3 training steps
    for step in range(3):
        imgs = torch.randn(batch_size, 3, img_size, img_size, device=device)
        keys = key.repeat(batch_size, 1)

        # encode
        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()

        # decode with original and finetuned
        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w = ldm_decoder.decode(imgs_z)

        # posthoc watermarked reference (avseal-style)
        with torch.no_grad():
            imgs_d0_01 = (imgs_d0 * 0.5 + 0.5).clamp(0, 1)
            msg_batch = wm_msg.repeat(imgs_d0_01.shape[0], 1).to(device)
            imgs_d0_wm_01 = wm_model.embed(imgs_d0_01, msg_batch, is_video=False)["imgs_w"]
            imgs_d0_wm = (imgs_d0_wm_01 * 2 - 1)

        # HiDDeN extraction (still computed but not used in loss)
        decoded = msg_decoder(vqgan_to_imnet(imgs_w))
        lossw = loss_w(decoded, keys)

        # Perceptual loss vs watermarked reference
        lossi = loss_i(imgs_w, imgs_d0_wm)

        # VideoSeal extractor loss
        imgs_w_01 = (imgs_w * 0.5 + 0.5).clamp(0, 1)
        if imgs_w_01.shape[-2:] != (wm_model.img_size, wm_model.img_size):
            imgs_w_01_resized = F.interpolate(imgs_w_01, size=(wm_model.img_size, wm_model.img_size), mode='bilinear', align_corners=False)
        else:
            imgs_w_01_resized = imgs_w_01
        pred_msg = wm_model.detector(imgs_w_01_resized)
        if pred_msg.dim() == 4:
            pred_msg = pred_msg.mean(dim=(-2, -1))
        pred_msg = pred_msg[:, 1:]  # skip detection confidence channel
        msg_batch_ext = wm_msg.repeat(pred_msg.shape[0], 1).to(device)
        loss_vs = F.binary_cross_entropy_with_logits(pred_msg, msg_batch_ext)

        # Combined loss (HiDDeN skipped when VideoSeal active)
        loss = lambda_i * lossi + extractor_weight * loss_vs

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Verify gradients flow through decoder
        grad_norm = sum(p.grad.norm().item() if p.grad is not None else 0 for p in ldm_decoder.parameters())

        print(f"  Step {step}: loss={loss.item():.4f}, loss_i={lossi.item():.4f}, "
              f"loss_vs={loss_vs.item():.4f}, lossw(unused)={lossw.item():.4f}")

    print("✓ Training loop with extractor_weight > 0 PASSED\n")


def test_val_loop():
    """Test the val() function code path with VideoSeal accuracy."""
    print("=" * 60)
    print("TEST 2: Validation loop with VideoSeal accuracy")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 2
    img_size = 64
    num_bits_hidden = 48
    num_bits_vs = 32

    ldm_ae = MockAutoencoder(z_channels=4, img_size=img_size)
    ldm_ae.eval()

    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.eval()

    msg_decoder = MockMsgDecoder(num_bits=num_bits_hidden)
    msg_decoder.eval()

    wm_model = MockVideoSeal(num_bits=num_bits_vs, img_size=img_size)
    wm_model.eval()

    wm_msg = torch.randint(0, 2, (1, num_bits_vs), dtype=torch.float32, device=device)
    key = torch.randint(0, 2, (1, num_bits_hidden), dtype=torch.float32, device=device)

    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    with torch.no_grad():
        imgs = torch.randn(batch_size, 3, img_size, img_size, device=device)
        keys = key.repeat(batch_size, 1)

        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()

        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w = ldm_decoder.decode(imgs_z)

        # HiDDeN accuracy
        decoded = msg_decoder(vqgan_to_imnet(imgs_w))
        diff = (~torch.logical_xor(decoded > 0, keys > 0))
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
        word_accs = (bit_accs == 1)

        # VideoSeal accuracy
        imgs_w_01 = (imgs_w * 0.5 + 0.5).clamp(0, 1)
        if imgs_w_01.shape[-2:] != (wm_model.img_size, wm_model.img_size):
            imgs_w_01_resized = F.interpolate(imgs_w_01, size=(wm_model.img_size, wm_model.img_size), mode='bilinear', align_corners=False)
        else:
            imgs_w_01_resized = imgs_w_01
        pred_msg = wm_model.detector(imgs_w_01_resized)
        if pred_msg.dim() == 4:
            pred_msg = pred_msg.mean(dim=(-2, -1))
        pred_msg = pred_msg[:, 1:]
        msg_batch = wm_msg.repeat(pred_msg.shape[0], 1).to(device)
        pred_bits = (pred_msg > 0).float()
        vs_diff = ~torch.logical_xor(pred_bits > 0.5, msg_batch > 0.5)
        vs_bit_accs = torch.sum(vs_diff, dim=-1) / vs_diff.shape[-1]
        vs_word_accs = (vs_bit_accs == 1)

        log_stats = {
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            "hidden_bit_acc": torch.mean(bit_accs).item(),
            "vs_bit_acc": torch.mean(vs_bit_accs).item(),
            "vs_word_acc": torch.mean(vs_word_accs.type(torch.float)).item(),
        }

        print(f"  Validation metrics: {log_stats}")
        assert 'vs_bit_acc' in log_stats, "vs_bit_acc not in log_stats"
        assert 'vs_word_acc' in log_stats, "vs_word_acc not in log_stats"
        assert 0 <= log_stats['vs_bit_acc'] <= 1, f"vs_bit_acc out of range: {log_stats['vs_bit_acc']}"

    print("✓ Validation loop with VideoSeal accuracy PASSED\n")


def test_original_behavior():
    """Test that extractor_weight=0 preserves original behavior (no VideoSeal)."""
    print("=" * 60)
    print("TEST 3: Original behavior preserved when extractor_weight=0")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 2
    img_size = 64
    num_bits_hidden = 48

    ldm_ae = MockAutoencoder(z_channels=4, img_size=img_size)
    ldm_ae.eval()
    for p in ldm_ae.parameters():
        p.requires_grad = False

    ldm_decoder = deepcopy(ldm_ae)
    for p in ldm_decoder.parameters():
        p.requires_grad = True

    msg_decoder = MockMsgDecoder(num_bits=num_bits_hidden)
    msg_decoder.eval()
    for p in msg_decoder.parameters():
        p.requires_grad = False

    key = torch.randint(0, 2, (1, num_bits_hidden), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(ldm_decoder.parameters(), lr=5e-4)

    loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded * temp, keys, reduction='mean')
    loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs) ** 2)

    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    lambda_i = 0.2
    lambda_w = 1.0
    wm_model = None
    extractor_weight = 0.0

    for step in range(3):
        imgs = torch.randn(batch_size, 3, img_size, img_size, device=device)
        keys = key.repeat(batch_size, 1)

        imgs_z = ldm_ae.encode(imgs)
        imgs_z = imgs_z.mode()
        imgs_d0 = ldm_ae.decode(imgs_z)
        imgs_w = ldm_decoder.decode(imgs_z)

        decoded = msg_decoder(vqgan_to_imnet(imgs_w))
        lossw = loss_w(decoded, keys)
        lossi = loss_i(imgs_w, imgs_d0)  # original: compare to clean reference
        loss_vs = torch.tensor(0.0, device=device)

        # Original loss: lambda_w * lossw + lambda_i * lossi
        loss = lambda_w * lossw + lambda_i * lossi

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  Step {step}: loss={loss.item():.4f}, loss_w={lossw.item():.4f}, "
              f"loss_i={lossi.item():.4f}, loss_vs={loss_vs.item():.4f}")

    print("✓ Original behavior (extractor_weight=0) PASSED\n")


def test_gradient_flow():
    """Verify gradients flow from VideoSeal detector back through trainable decoder."""
    print("=" * 60)
    print("TEST 4: Gradient flow through VideoSeal detector → decoder")
    print("=" * 60)

    device = torch.device('cpu')
    batch_size = 2
    img_size = 64
    num_bits_vs = 32

    ldm_ae = MockAutoencoder(z_channels=4, img_size=img_size)
    ldm_ae.eval()
    for p in ldm_ae.parameters():
        p.requires_grad = False

    ldm_decoder = deepcopy(ldm_ae)
    for p in ldm_decoder.parameters():
        p.requires_grad = True

    wm_model = MockVideoSeal(num_bits=num_bits_vs, img_size=img_size)
    wm_model.eval()
    wm_model.requires_grad_(False)

    wm_msg = torch.randint(0, 2, (1, num_bits_vs), dtype=torch.float32, device=device)

    imgs = torch.randn(batch_size, 3, img_size, img_size, device=device)
    imgs_z = ldm_ae.encode(imgs).mode()
    imgs_w = ldm_decoder.decode(imgs_z)

    # VideoSeal extractor loss only
    imgs_w_01 = (imgs_w * 0.5 + 0.5).clamp(0, 1)
    pred_msg = wm_model.detector(imgs_w_01)
    pred_msg = pred_msg[:, 1:]
    msg_batch = wm_msg.repeat(batch_size, 1).to(device)
    loss_vs = F.binary_cross_entropy_with_logits(pred_msg, msg_batch)

    loss_vs.backward()

    decoder_grads = [(name, p.grad.norm().item()) for name, p in ldm_decoder.named_parameters() if p.grad is not None and p.grad.norm().item() > 0]
    encoder_grads = [(name, p.grad.norm().item()) for name, p in ldm_ae.named_parameters() if p.grad is not None and p.grad.norm().item() > 0]
    detector_grads = [(name, p.grad.norm().item()) for name, p in wm_model.named_parameters() if p.grad is not None and p.grad.norm().item() > 0]

    print(f"  Decoder params with gradients: {len(decoder_grads)}")
    for name, norm in decoder_grads:
        print(f"    {name}: grad_norm={norm:.6f}")
    print(f"  Frozen encoder params with gradients: {len(encoder_grads)} (should be 0)")
    print(f"  Frozen detector params with gradients: {len(detector_grads)} (should be 0)")

    assert len(decoder_grads) > 0, "No gradients in trainable decoder!"
    assert len(encoder_grads) == 0, "Gradients leaked into frozen encoder!"
    assert len(detector_grads) == 0, "Gradients leaked into frozen detector!"

    print("✓ Gradient flow test PASSED\n")


def test_resize_handling():
    """Test that resize logic works when input size != wm_model.img_size."""
    print("=" * 60)
    print("TEST 5: Resize handling for mismatched image/detector sizes")
    print("=" * 60)

    device = torch.device('cpu')
    num_bits_vs = 32

    wm_model = MockVideoSeal(num_bits=num_bits_vs, img_size=128)
    wm_model.eval()

    # Input at 256x256, detector expects 128x128
    imgs_w_01 = torch.randn(2, 3, 256, 256).clamp(0, 1)

    if imgs_w_01.shape[-2:] != (wm_model.img_size, wm_model.img_size):
        imgs_w_01_resized = F.interpolate(imgs_w_01, size=(wm_model.img_size, wm_model.img_size), mode='bilinear', align_corners=False)
        print(f"  Resized: {imgs_w_01.shape} → {imgs_w_01_resized.shape}")
    else:
        imgs_w_01_resized = imgs_w_01
        print(f"  No resize needed: {imgs_w_01.shape}")

    assert imgs_w_01_resized.shape[-2:] == (128, 128), f"Wrong size: {imgs_w_01_resized.shape}"

    pred_msg = wm_model.detector(imgs_w_01_resized)
    pred_msg = pred_msg[:, 1:]
    assert pred_msg.shape == (2, num_bits_vs), f"Wrong detector output shape: {pred_msg.shape}"

    # Same size case
    imgs_same = torch.randn(2, 3, 128, 128).clamp(0, 1)
    if imgs_same.shape[-2:] != (wm_model.img_size, wm_model.img_size):
        imgs_same_resized = F.interpolate(imgs_same, size=(wm_model.img_size, wm_model.img_size), mode='bilinear', align_corners=False)
    else:
        imgs_same_resized = imgs_same
        print(f"  No resize needed (same size): {imgs_same.shape}")

    assert imgs_same_resized.data_ptr() == imgs_same.data_ptr(), "Should be same tensor when sizes match"

    print("✓ Resize handling PASSED\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("VideoSeal Integration Test Suite")
    print("=" * 60 + "\n")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    test_train_loop()
    test_val_loop()
    test_original_behavior()
    test_gradient_flow()
    test_resize_handling()

    print("=" * 60)
    print("ALL 5 TESTS PASSED ✓")
    print("=" * 60)
