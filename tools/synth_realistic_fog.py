#!/usr/bin/env python3
"""
synth_realistic_fog.py

生成更真实的合成雾（支持 uniform / depth-linear / radial 等），并可选加入云状 noise 与远处模糊。

主要修正：
 - 在合成时使用线性 RGB（sRGB <-> linear）进行模糊与混合，避免伽马非线性导致的色偏（例如偏紫）
 - 在模糊时保持 float32（不先转 uint8），避免量化误差
 - 缩小大气光 A 的通道抖动默认值，减少随机偏色
 - 接受 --blur 作为 blur_strength 别名（方便 CLI 使用）
"""

import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np

# ----------------------------
# Helpers: dirs / io
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_image_rgb(p: Path):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb

def save_rgb_png(path: Path, rgb_uint8: np.ndarray):
    # cv2 expects BGR
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])

def save_tmap(path: Path, t: np.ndarray):
    # t float in [0,1] -> 0..255 uint8
    t8 = (np.clip(t, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), t8, [cv2.IMWRITE_PNG_COMPRESSION, 3])

# ----------------------------
# sRGB <-> Linear helpers
# ----------------------------
def srgb_to_linear(img):
    """
    img: float image in [0,1]
    returns linear RGB in [0,1]
    """
    img = img.astype(np.float32)
    a = 0.055
    mask = img <= 0.04045
    out = np.empty_like(img, dtype=np.float32)
    out[mask] = img[mask] / 12.92
    out[~mask] = ((img[~mask] + a) / (1.0 + a)) ** 2.4
    return out

def linear_to_srgb(img):
    """
    img: linear RGB in [0,1]
    returns sRGB in [0,1]
    """
    img = img.astype(np.float32)
    a = 0.055
    mask = img <= 0.0031308
    out = np.empty_like(img, dtype=np.float32)
    out[mask] = img[mask] * 12.92
    out[~mask] = (1.0 + a) * (img[~mask] ** (1.0 / 2.4)) - a
    return np.clip(out, 0.0, 1.0)

# ----------------------------
# Noise / texture for cloud-like fog
# ----------------------------
def smooth_noise(h, w, scale=0.05, octaves=3, base_seed=None):
    """
    Create smooth noise map in [-1,1], combining multiple Gaussian-blurred noises (approx Perlin-like).
    """
    rng = np.random.RandomState(base_seed) if base_seed is not None else np.random
    total = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    freq = 1.0
    amp_sum = 0.0
    for o in range(octaves):
        small_h = max(4, int(h / freq))
        small_w = max(4, int(w / freq))
        noise = rng.normal(loc=0.0, scale=1.0, size=(small_h, small_w)).astype(np.float32)
        up = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        k = max(3, int(min(h, w) / (8.0 * freq)) | 1)
        up = cv2.GaussianBlur(up, (k, k), sigmaX=0, sigmaY=0)
        total += amplitude * up
        amp_sum += amplitude
        amplitude *= scale
        freq *= 2.0
    if amp_sum > 0:
        total /= amp_sum
    mn, mx = total.min(), total.max()
    if mx - mn > 1e-6:
        total = 2.0 * (total - mn) / (mx - mn) - 1.0
    else:
        total = np.zeros_like(total)
    return total

# ----------------------------
# Fog generation kernels
# ----------------------------
def make_vertical_depth_map(h, w, invert=False):
    """Return depth-like map in [0,1], 0=near (bottom), 1=far (top) by default."""
    if invert:
        ys = np.linspace(0.0, 1.0, h)
    else:
        ys = np.linspace(1.0, 0.0, h)
    grid = np.repeat(ys[:, None], w, axis=1).astype(np.float32)
    return grid

def make_radial_depth_map(h, w, cx=None, cy=None):
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0
    xs = np.arange(w).astype(np.float32)
    ys = np.arange(h).astype(np.float32)
    X, Y = np.meshgrid(xs, ys)
    d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return d

# ----------------------------
# Main fog synthesis (physical model)
# ----------------------------
def synthesize_fog(J_rgb_uint8, A_rgb=(1.0,1.0,1.0), t_map=None, beta=1.0, depth_map=None,
                   noise_level=0.0, noise_octaves=3, noise_scale=0.05, blur_kernel_for_noise=101, seed=None,
                   apply_blur_far=True, blur_strength=21):
    """
    Synthesize foggy image given clean image J (RGB uint8).
    The blending & blur are performed in linear RGB to avoid gamma-related color bias.
    """
    assert J_rgb_uint8.ndim == 3 and J_rgb_uint8.shape[2] == 3
    h, w = J_rgb_uint8.shape[:2]
    # convert J to float [0,1]
    J_srgb = (J_rgb_uint8.astype(np.float32) / 255.0).astype(np.float32)

    if t_map is None:
        if depth_map is None:
            depth_map = np.ones((h, w), dtype=np.float32) * 0.5
        else:
            depth_map = np.clip(depth_map.astype(np.float32), 0.0, 1.0)
        if noise_level > 0.0:
            n = smooth_noise(h, w, scale=noise_scale, octaves=noise_octaves, base_seed=seed)
            depth_mod = depth_map * (1.0 + noise_level * n)
            depth_mod = np.clip(depth_mod, 0.0, 1.0)
            depth_used = depth_mod
        else:
            depth_used = depth_map
        t = np.exp(-beta * depth_used).astype(np.float32)
        t = np.clip(t, 0.0, 1.0)
    else:
        t = np.clip(t_map.astype(np.float32), 0.0, 1.0)

    # convert to linear RGB for physically-correct blending
    J_lin = srgb_to_linear(J_srgb)

    # prepare atmospheric light in linear domain
    A_arr = np.array(A_rgb, dtype=np.float32).reshape(1,1,3)
    A_lin = srgb_to_linear(A_arr)

    # optionally simulate far blur: blur on linear float image (avoid uint8 quantize)
    if apply_blur_far and (blur_strength is not None) and int(blur_strength) >= 3:
        k = int(blur_strength) | 1
        # OpenCV supports float32 for GaussianBlur
        J_blur_lin = cv2.GaussianBlur(J_lin, (k, k), sigmaX=0, sigmaY=0)
        blur_weight = (1.0 - t) ** 1.0
        bw3 = np.repeat(blur_weight[:, :, None], 3, axis=2)
        J_effective_lin = J_lin * (1.0 - bw3) + J_blur_lin * bw3
    else:
        J_effective_lin = J_lin

    # linear blending
    t3 = np.repeat(t[:, :, None], 3, axis=2)
    I_lin = J_effective_lin * t3 + A_lin * (1.0 - t3)
    I_lin = np.clip(I_lin, 0.0, 1.0)

    # convert back to sRGB and uint8 for saving/display
    I_srgb = linear_to_srgb(I_lin)
    I_uint8 = (I_srgb * 255.0).astype(np.uint8)
    return I_uint8, t

# ----------------------------
# Sample A (atmospheric light)
# ----------------------------
def sample_A(minA=0.7, maxA=1.0, color_bias=(0.0, 0.0, 0.0), jitter=0.01, rng=None):
    """
    Sample near-neutral atmospheric light. jitter default small (0.01) to avoid strong color cast.
    """
    if rng is None:
        rng = random
    base = rng.uniform(minA, maxA)
    A = []
    for c in range(3):
        a = base + rng.uniform(-jitter, jitter) + color_bias[c]
        A.append(float(np.clip(a, 0.0, 1.0)))
    return tuple(A)

# ----------------------------
# Main processing loop
# ----------------------------
def process_folder(images_dir, out_dir, mode='depth', variants=1,
                   beta_min=0.6, beta_max=1.6, noise=0.4, noise_octaves=3, noise_scale=0.05,
                   blur_noise_ksize=101, minA=0.75, maxA=1.0, A_jitter=0.01,
                   apply_blur_far=True, blur_strength=21,
                   radial_center_jitter=0.15, seed=None, overwrite=False):
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')])
    if len(files) == 0:
        raise RuntimeError(f"No images in {images_dir}")

    rng = random.Random(seed)
    metadata = {}

    for p in files:
        img_rgb = read_image_rgb(p)
        if img_rgb is None:
            print("WARN: failed to read", p)
            continue
        h,w = img_rgb.shape[:2]
        metadata[p.name] = []
        for vi in range(variants):
            beta = float(rng.uniform(beta_min, beta_max))
            A = sample_A(minA, maxA, color_bias=(0.0,0.0,0.0), jitter=A_jitter, rng=rng)
            if mode == 'uniform':
                depth_map = np.ones((h,w), dtype=np.float32) * rng.uniform(0.3, 0.9)
            elif mode == 'depth':
                depth_map = make_vertical_depth_map(h,w, invert=False)
                depth_map = np.clip(depth_map * rng.uniform(0.8,1.2), 0.0, 1.0)
            elif mode == 'radial':
                cx = w*(0.5 + rng.uniform(-radial_center_jitter, radial_center_jitter))
                cy = h*(0.5 + rng.uniform(-radial_center_jitter, radial_center_jitter))
                depth_map = make_radial_depth_map(h,w,cx=cx,cy=cy)
            else:
                raise ValueError("Unsupported mode: "+str(mode))

            _seed = rng.randint(0,2**31-1)
            I_fog, t_map = synthesize_fog(
                img_rgb,
                A_rgb=A,
                depth_map=depth_map,
                beta=beta,
                noise_level=float(noise),
                noise_octaves=int(noise_octaves),
                noise_scale=float(noise_scale),
                blur_kernel_for_noise=int(blur_noise_ksize),
                seed=_seed,
                apply_blur_far=bool(apply_blur_far),
                blur_strength=int(blur_strength)
            )

            stem = p.stem
            out_img_p = out_dir / f"{stem}.png"
            out_t_p   = out_dir / f"{stem}_t_v{vi+1}.png"
            if out_img_p.exists() and not overwrite:
                print(f"Skip exists {out_img_p}")
                continue
            save_rgb_png(out_img_p, I_fog)
            save_tmap(out_t_p, t_map)
            meta_item = {
                'source': str(p),
                'out_image': str(out_img_p),
                'out_tmap': str(out_t_p),
                'mode': mode,
                'variant': vi+1,
                'beta': beta,
                'A': list(A),
                'noise': float(noise),
                'noise_octaves': int(noise_octaves),
                'noise_scale': float(noise_scale),
                'blur_strength': int(blur_strength),
                'seed': int(_seed)
            }
            metadata[p.name].append(meta_item)
            print(f"Saved {out_img_p.name} beta={beta:.3f} A={A} seed={_seed}")
    meta_p = out_dir / "metadata.json"
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print("Wrote metadata:", meta_p)

# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Synthesize more realistic foggy images")
    p.add_argument('--images', '-i', default='datasets/images', help='input images dir')
    p.add_argument('--out', '-o',default='datasets/fog_images', help='output dir for fog images')
    p.add_argument('--mode', choices=['uniform','depth','radial'], default='uniform', help='fog spatial mode')
    p.add_argument('--variants', type=int, default=2, help='how many variants per input image')
    p.add_argument('--beta-min', type=float, default=0.6)
    p.add_argument('--beta-max', type=float, default=1.6)
    p.add_argument('--noise', type=float, default=0, help='smooth noise amplitude (0..1)')
    p.add_argument('--noise-octaves', type=int, default=3)
    p.add_argument('--noise-scale', type=float, default=0.05)
    p.add_argument('--blur-noise-ksize', type=int, default=101, help='unused in current implementation but kept for backward compat')
    p.add_argument('--minA', type=float, default=0.75)
    p.add_argument('--maxA', type=float, default=1.0)
    p.add_argument('--A-jitter', type=float, default=0.01)
    p.add_argument('--apply-blur-far', action='store_true', help='apply extra blur in far regions')
    # accept --blur as alias for blur_strength to match your example
    p.add_argument('--blur', type=int, default=21, help='alias for --blur-strength (kernel size for far blur, odd int)')
    p.add_argument('--blur-strength', type=int, default=None, help='kernel size for far blur (odd int)')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    # prefer explicit blur_strength if provided
    blur_strength = args.blur_strength if args.blur_strength is not None else args.blur
    # ensure odd and at least 3
    blur_strength = max(3, int(blur_strength) | 1)
    t0 = time.time()
    process_folder(
        args.images,
        args.out,
        mode=args.mode,
        variants=args.variants,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        noise=args.noise,
        noise_octaves=args.noise_octaves,
        noise_scale=args.noise_scale,
        blur_noise_ksize=args.blur_noise_ksize,
        minA=args.minA,
        maxA=args.maxA,
        A_jitter=args.A_jitter,
        apply_blur_far=args.apply_blur_far,
        blur_strength=blur_strength,
        seed=args.seed,
        overwrite=args.overwrite
    )
    print("Done in %.1f s" % (time.time()-t0))

if __name__ == '__main__':
    main()
