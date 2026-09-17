<div align="center">

# ColorAwareNet

**Color-gain-guided image dehazing and semantic segmentation**

**English** · [简体中文](README.zh-CN.md)

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0_GPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](#installation)
[![Training](https://img.shields.io/badge/Entry-train.py-0F766E?style=flat-square)](train.py)

[Training entrypoint](train.py) · [Dataset](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888) · [GitHub Issues](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues)

</div>

> **Paper** · *A Color-Gain-Guided, Color-Preserving Joint Framework for Image Dehazing and Semantic Segmentation*

**Navigate:** [Overview](#overview) · [Quick start](#quick-start) · [Installation](#installation) · [Data](#datasets) · [Method](#method) · [Training](#training) · [Evaluation](#prediction-and-evaluation) · [Checkpoints](#checkpoints) · [Experiments](#ablations-and-visualization) · [FAQ](#faq) · [Citation](#citation)

## Overview

**ColorAwareNet couples image dehazing with road segmentation in a single trainable pipeline.** This repository provides the PyTorch implementation of the paper above. Given a hazy RGB image, **ColorAwareUNet** produces a restored image, and **LiteAttentionUNet** predicts a road/background mask from that result.

The paper asks how restoration can improve visibility and preserve color while retaining structures useful for semantic segmentation. Haze reduces contrast and weakens object boundaries; restoration artifacts can also change the visual cues used to recognize a road. The framework brings restoration and segmentation objectives into the same training process, with image quality, color consistency and semantic accuracy evaluated together.

[![Figure 2 from the paper: overall architecture, training workflow and inference process of the proposed joint framework.](docs/assets/framework.png)](docs/assets/framework.png)

*Figure 2 from the manuscript: overall architecture, training workflow and inference process. Click the image to view the original resolution.*

### Key ideas

- **An explicit RGB color gain.** The dehazer predicts three gain values per image, one for each color channel. Each value is applied across the image, providing a compact correction whose effect on color can be inspected directly.
- **Local detail reconstruction.** A spatial residual and a refinement head complement the global gain, allowing corrections to vary across regions and recover structures such as road boundaries and markings.
- **Segmentation-aware restoration.** A lightweight attention segmenter learns from the restored images. After separate pretraining stages, joint fine-tuning lets the segmentation loss also update the dehazer, alongside its image-restoration objective.

At inference, **only a hazy image is required**. Clear reference images and road masks provide supervision during training and are used for paired evaluation.

### What you can do with this repository

| Workflow | Data / configuration | Main outputs |
| :--- | :--- | :--- |
| [Train the joint model](#training) | Aligned road hazy/clear/mask triplets | Three-stage checkpoints, saved data split and training curves |
| [Run dehazing experiments](#datasets) | SOTS indoor/outdoor and augmented HSTS; five retained baselines | Restored images and image-quality/color metrics |
| [Predict and evaluate](#prediction-and-evaluation) | A single image, a folder or a paired validation set | Restored RGB, road masks, overlays and metric summaries |
| [Study the components](#ablations-and-visualization) | Registered architecture, gain, attention and training ablations | Experiment summaries, gain visualizations and component figures |

The road experiment uses **185 paired samples**, split into **157 training / 28 validation** samples, with no independent test set. SOTS and augmented HSTS provide dehazing evaluation without semantic labels. See [datasets](#datasets) and [evaluation](#prediction-and-evaluation) for the pairing rules and reporting protocols.

This release contains source code, documentation and the manuscript's network architecture figures. Prepare datasets and checkpoints locally; the full manuscript and notebooks are excluded.

---

## Quick start

**The training entrypoint is [`train.py`](train.py).** The shared training loop is in [`dehaze_seg/engine/train.py`](dehaze_seg/engine/train.py).

1. Complete [installation](#installation) and run all commands from the repository root.
2. Download [datasets.zip](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888), using extraction code **8888**, and prepare the [dataset layout](#datasets).
3. Start the paper's default 60 / 20 / 20 training schedule:

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --device cuda --amp
```

After training, predict an image using the generated checkpoint:

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
```

Replace `001.png` with your image filename. The training examples explicitly select a CUDA GPU with `--device cuda --amp`; prediction selects an available GPU automatically. For a smaller CPU workflow check, see [training](#training). The default perceptual loss needs pretrained VGG16 weights; see [FAQ](#faq) for offline caching.

## Installation

Clone the repository and run subsequent commands from its root:

```bash
git clone https://github.com/dongfangshiwen/colorawarenet-joint-model.git
cd colorawarenet-joint-model
```

Use Python **3.12** to match the paper environment (the package requires Python 3.10+), and create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux **before installing dependencies**.

### GPU installation

Section 4.1.3 of the manuscript reports **PyTorch 2.3.0 and CUDA 12.1**. For an NVIDIA GPU with a compatible driver, install the CUDA build with its matching torchvision version from the [official PyTorch version table](https://pytorch.org/get-started/previous-versions/#v230):

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e . "numpy<2"
python -m dehaze_seg --help
```

The NumPy constraint keeps this older PyTorch environment on NumPy 1.x to avoid binary compatibility problems; it is an installation constraint, not a NumPy version reported in the manuscript. See [NumPy's compatibility guidance](https://numpy.org/doc/stable/user/troubleshooting-importerror.html#downstream-importerror-attributeerror-or-c-api-abi-incompatibility).

Check GPU visibility before starting training:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
```

`--device cuda` requires a visible CUDA GPU; `--amp` enables GPU mixed precision. The CLI default `--device auto` selects CUDA when available. Installation also provides the `dehaze-seg` executable.

<details>
<summary><strong>CPU installation for local workflow checks</strong></summary>

Use a separate virtual environment for the locally verified CPU combination:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Run the smaller example under [training](#training) with `--device cpu`. This is the local validation environment; the paper experiments used a GPU.

</details>

<details>
<summary><strong>Paper environment and local validation</strong></summary>

| Environment | Configuration |
| :--- | :--- |
| Paper software environment | Ubuntu 22.04 · Python 3.12 · PyTorch 2.3.0 · CUDA 12.1 |
| Paper cloud hardware | One 32 GB vGPU · 16 Intel Xeon Platinum 8352V vCPUs · 62 GB RAM |
| Local CPU regression checks | Windows · Python 3.12 · PyTorch 2.8.0+cpu · torchvision 0.23.0 |

The GPU configuration above is taken from the manuscript. The checks performed during this repository cleanup used the local CPU environment; CUDA execution and complete experimental reproduction were not revalidated here. Dependency ranges do not mean every version combination was tested.

</details>

## Datasets

The paired road dataset contains **185 hazy/clear/mask triplets**:

- Archive: **datasets.zip**
- [Baidu Netdisk download](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888)
- Extraction code: **8888**

Download and extract manually:

```text
datasets/
├── hazy/       # 001.png, 002.png, ...
├── clear/      # Matching clear images
└── masks/      # Matching segmentation masks
```

`--data-root` must contain `hazy/`, `clear/` and `masks/`, with matching file stems such as `001.png`. File extensions may differ. Images are RGB; binary masks use background=0 and road=1, with 0/255 grayscale masks also supported. Masks use nearest-neighbor resizing and share geometric transforms with their images.

New road experiments group byte-identical clear reference files before splitting, using seed 42 and validation ratio 0.15. The checked 185-sample dataset still yields **157 training and 28 validation samples**, with **no independent test split**. Whole groups are kept together; on other datasets the actual count may differ. Training saves the exact IDs, grouping and hashes in `split.json` and the checkpoint.

The local data contains five duplicate-reference pairs: `058/114`, `062/130`, `063/140`, `177/179`, `181/182`. The historical sample-ID split placed `114` and `181` across the training/validation boundary from their paired reference. The corrected scene split changes the sample membership, so it requires new training and must not be claimed to reproduce the original split or scores. `--split-unit sample` retains the old ID-based split for historical investigation and records its reference overlap. Existing weights retain their saved split; do not evaluate old weights on a newly generated split and assume independence.

### Benchmark datasets

Other datasets must be prepared separately:

| Dataset option | Required layout and pairing |
|---|---|
| `sots-indoor`, `sots-outdoor` | `hazy/` and `clear/` or `gt/`; exact stem first, then scene prefix (`1400_1` → `1400`) |
| `hsts` | `synthetic/synthetic/` for haze, `synthetic/original/` for matching clear images |

These benchmarks use restoration-only supervision. No dummy segmentation masks are generated. HSTS keeps paired crop, flip and rotation augmentation; `--train-repeats` controls repeated training samples. Results on augmented HSTS-derived data are not results under the unmodified official HSTS protocol. Unpaired real images can be processed with `predict --input`, without full-reference evaluation.

## Method

### From a hazy image to a road mask

**1. Restore color and spatial detail.** ColorAwareUNet uses a U-Net encoder–decoder with skip connections. Features from its bottleneck predict the global RGB gain, while the decoder predicts a residual at image resolution. A refinement head then combines the input, coarse restoration, residual and decoder features to produce a final correction. With the main configuration:

```text
gain     = max(0.95, 1 + 0.30 × tanh(raw_gain))
coarse   = hazy × gain + 0.50 × residual
restored = clamp(coarse + 0.25 × refinement, 0, 1)
```

Here, `gain` contains three values shared across spatial positions; `residual` and `refinement` are three-channel maps at image resolution. The gain head learns from each input, so its values can differ between images.

**2. Segment the restored image.** ImageNet normalization prepares the restored RGB image for LiteAttentionUNet. Depthwise-separable convolutions provide lightweight feature extraction, and attention gates use decoder context to filter encoder skip features. The main configuration enables attention and disables SE and the auxiliary head. The final classifier assigns each pixel to road or background.

**3. Learn the two tasks progressively.** Dehazing pretraining establishes the restoration mapping. The dehazer is then frozen while the segmenter learns from its outputs. In joint fine-tuning, restoration and segmentation losses update both networks through the connected pipeline. The [60 / 20 / 20 schedule](#training) specifies which parameters are updated at each stage.

<details>
<summary><strong>Network architecture figures from the manuscript</strong></summary>

**Figure 3 · ColorAwareUNet.** Global RGB gain, residual prediction and refinement modules.

[![Figure 3 from the paper: ColorAwareUNet architecture.](docs/assets/colorawareunet.png)](docs/assets/colorawareunet.png)

**Figure 4 · LiteAttentionUNet.** Lightweight convolutions, attention gates and optional SE blocks.

[![Figure 4 from the paper: LiteAttentionUNet architecture.](docs/assets/liteattentionunet.png)](docs/assets/liteattentionunet.png)

These are the original figures embedded in the supplied manuscript. Click either image to view it at full resolution. The default settings for the optional modules are listed below.

</details>

### Supported models

| `--model` | Implementation |
| :--- | :--- |
| `coloraware` | **ColorAwareUNet**, the paper's main dehazer |
| `c2pnet` | C2PNet implementation in this repository |
| `dcp` | Trainable DCP recovery/refinement + LiteAttentionUNet for joint training; classical DCP remains available for checkpoint-free inference |
| `ffanet` | FFA-Net implementation in this repository |
| `grid` | GridDehazeNet implementation in this repository |
| `psd` | PSDDehazeNet implementation in this repository |

All road experiments use **LiteAttentionUNet** for segmentation. Section 4.1.5 of the manuscript evaluates different dehazers using **one shared, frozen LiteAttentionUNet checkpoint**, without method-specific segmentation fine-tuning. Use `--segmenter-checkpoint` for this protocol. Neural baselines retain this repository's implementations; equivalence to the original authors' code, weights or scores is not claimed. The public name is always `dcp`: training defaults to a learned refinement extension, checkpoint-free inference uses classical recovery, and historical checkpoints retain their original implementation. Results distinguish `learned-refinement`, `classical` and `historical-enhanced`. Report the trainable extension as a learned DCP variant when using it in the paper.

### Paper configuration

| Component | Default |
| :--- | :--- |
| Dehazer base channels | 32 |
| RGB gain | Global · tanh · scale **0.30** · minimum **0.95** · effective range [0.95, 1.30] |
| Residual / refinement scale | **0.50 / 0.25** |
| Segmenter | 2 classes · base channels 32 · width multiplier 1.0 |
| Attention / SE / auxiliary head | On / off / off |
| ImageNet input normalization | On |
| Training resolution / batch size | **512 × 512 / 2** |
| Optimizer | AdamW · learning rate 1e-4 · weight decay 1e-5 |

### Training objectives

```text
Restoration:   L1 + 0.40 × SSIM_loss + 0.05 × VGG_perceptual
Segmentation: CE + soft Dice
Joint:        restoration + segmentation
```

`SSIM_loss = (1 − SSIM) / 2` uses an 11×11 averaging window. Perceptual loss uses frozen ImageNet VGG16 feature indices 3, 8 and 15. Soft Dice is averaged over classes; both task weights default to 1 during joint fine-tuning.

MSE, gradient discrepancy, saturation deviation (ΔSat) and chromaticity ratio error (CRerr) are monitored without contributing to the default loss. Side outputs are retained for visualization and checkpoint compatibility, without auxiliary supervision. Local and amplify-only gain are ablation settings. The global gain contribution heatmap displays `mean(abs(input × (gain−1)))`; its spatial variation comes from the input, while the gain itself is global.

### Paper components in the code

| Component | Source |
| :--- | :--- |
| RGB gain, residual and refinement | [`ColorAwareUnet.py`](dehaze_seg/models/ColorAwareUnet.py) |
| Attention segmentation | [`LiteAttentionUnet.py`](dehaze_seg/models/LiteAttentionUnet.py) |
| Joint network and output adaptation | [`joint.py`](dehaze_seg/models/joint.py) · [`registry.py`](dehaze_seg/models/registry.py) |
| Training objectives | [`losses.py`](dehaze_seg/losses.py) |
| Three-stage optimization | [`engine/train.py`](dehaze_seg/engine/train.py) |
| Controlled experiments | [`ablations.py`](dehaze_seg/experiments/ablations.py) |

## Training

`python train.py` accepts the same options as `python -m dehaze_seg train`. Run `python train.py --help` for the complete argument list.

### Three-stage schedule

| Stage | Epochs | Updated parameters | Objective |
| :--- | :---: | :--- | :--- |
| 1 · Dehazing pretraining | **60** | ColorAwareUNet | Restoration loss |
| 2 · Segmentation training | **20** | LiteAttentionUNet; dehazer frozen | Segmentation loss |
| 3 · Joint fine-tuning | **20** | Both networks | Both losses |

### Joint road training

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --device cuda --amp
```

<details>
<summary><strong>Explicit paper parameters</strong></summary>

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --pretrain-dehaze-epochs 60 --pretrain-seg-epochs 20 --finetune-epochs 20 --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/paper-explicit --device cuda --amp
```

</details>

<details>
<summary><strong>Smaller CPU workflow check</strong></summary>

This runs all three stages at 64×64 and disables perceptual loss to avoid downloading VGG weights. It processes the prepared dataset and creates separate output. This altered setup is for checking the workflow, not reproducing paper scores.

```bash
python train.py --data-root datasets --device cpu --resize 64 64 --batch-size 2 --pretrain-dehaze-epochs 1 --pretrain-seg-epochs 1 --finetune-epochs 1 --lam-perc 0 --output runs/smoke --no-plots
```

</details>

### Baselines

Train a neural dehazing baseline on the same road split, then evaluate it with the shared frozen segmenter:

```bash
python train.py --dataset paired-road --data-root datasets --model c2pnet --pretrain-seg-epochs 0 --finetune-epochs 0 --output runs/comparison --device cuda --amp
```

This command updates only the dehazer. Its checkpoint's untrained segmenter must be replaced using `--segmenter-checkpoint` in the comparison below. Omitting the two zero-epoch options enables a separate joint-training experiment.

### Joint DCP dehazing and segmentation training

`train.py --model dcp` now trains both the **dehazing refinement** and **LiteAttentionUNet** on `paired-road` triplets (`hazy/`, `clear/`, `masks/`). The default `--dcp-mode learned` uses the repository's analytic DCP recovery with bounded restoration, a learned detail-refinement head and detail/sharpening processing. The analytic prior has no learned parameters; the refinement head has 22,563 trainable parameters and is part of the dehazing branch.

```bash
python train.py --model dcp --dataset paired-road --data-root datasets --output runs/dcp-joint --device cuda --amp
```

| Stage | Default epochs | Updated branch | Objective |
| :--- | :---: | :--- | :--- |
| `dehaze` | 60 | DCP learned refinement | L1 + 0.40 × SSIM loss + 0.05 × VGG perceptual loss |
| `seg` | 20 | LiteAttentionUNet; dehazer frozen | CE + Dice |
| `joint` | 20 | Both trainable branches | Restoration + segmentation; segmentation gradients reach the dehazer |

The epoch options retain their normal meanings: `--pretrain-dehaze-epochs`, `--pretrain-seg-epochs`, `--finetune-epochs`. VGG weights are required for the default restoration loss. SOTS-style data and HSTS use restoration-only training with `--dcp-mode learned`, without segmentation masks; keep training and test data separate as described below.

Weights are saved under `runs/dcp-joint/paired-road/dcp/`, including `dehaze/`, `seg/`, `joint/` stage checkpoints and top-level `best.pth` / `last.pth`. Model selection uses the same stage-specific scores as the main trainer. The configuration records `dehazer_type=learned-dcp`, the refinement settings and the actual schedule, so loading restores the trainable architecture automatically.

```bash
python -m dehaze_seg predict --checkpoint runs/dcp-joint/paired-road/dcp/best.pth --input datasets/hazy/001.png --output results/dcp-joint-predict
python -m dehaze_seg evaluate --checkpoint runs/dcp-joint/paired-road/dcp/best.pth --dataset paired-road --data-root datasets --split val --output results/dcp-joint-eval
```

The previous frozen-DCP segmentation workflow remains available explicitly:

```bash
python train.py --model dcp --dcp-mode classical --dataset paired-road --data-root datasets --output runs/dcp-seg --device cuda --amp
```

In this optional mode only the segmenter trains, for `pretrain-seg-epochs + finetune-epochs` (40 by default); restoration pretraining is unused and VGG is not needed. Old checkpoints from this mode continue to load as fixed classical DCP. New joint runs require a fresh output directory and retraining.

Joint training is an additional learned-DCP experiment. For the manuscript's shared-segmenter comparison, pass the **same** `--segmenter-checkpoint` for every dehazer as shown below; independently trained per-method segmenters define a different experiment.

<details>
<summary><strong>Neural dehazer training: SOTS-style data and augmented HSTS</strong></summary>

```bash
python train.py --dataset sots-indoor --data-root datasets/ITS/train --val-root datasets/ITS/val --model coloraware --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/sots --device cuda --amp
python train.py --dataset sots-outdoor --data-root datasets/OTS/train --val-root datasets/OTS/val --model coloraware --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/sots --device cuda --amp
python train.py --dataset hsts --data-root HSTS --train-repeats 4 --output runs/hsts --device cuda --amp
```

Prepare independent ITS/OTS training and validation directories with `hazy/` and `clear/` or `gt/`; these paths are layout examples, not supplied official splits. Keep SOTS test data in `SOTS/indoor` and `SOTS/outdoor`, outside both training and model selection. `--dataset sots-*` selects the pairing adapter, not permission to train on SOTS test images. Without `--val-root`, SOTS-style data is split by the SHA-256 of each paired clear reference, keeping all haze variants of a scene together. The validation fraction then applies to scene groups.

Benchmark training runs only `--pretrain-dehaze-epochs`, with restoration supervision. The examples use 512×512 inputs, learning rate 1e-4, batch size 2 and 60 restoration epochs. The HSTS command is an internal augmented experiment; evaluate its saved validation split, not all of the training directory. `--train-repeats 4` is a configurable example, not an official HSTS protocol.

</details>

### Common options

| Option | Purpose / default |
| :--- | :--- |
| `--pretrain-dehaze-epochs`, `--pretrain-seg-epochs`, `--finetune-epochs` | Stage lengths: 60 / 20 / 20 |
| `--resize H W`, `--batch-size` | Input size and batch size: 512 512 / 2 |
| `--seed`, `--val-ratio` | Split seed and validation fraction: 42 / 0.15 |
| `--val-root` | Independent validation directory; all training-root samples are used for training |
| `--split-unit` | Default: `scene` for roads/SOTS, `sample` for HSTS; `sample` selects historical ID-based splitting |
| `--device`, `--amp` | `auto`, `cpu` or `cuda`; opt-in CUDA mixed precision |
| `--workers`, `--threads` | Loader workers: 0; CPU threads: 4 |
| `--use-se`, `--no-attention`, `--no-imagenet-norm` | Change the segmenter's default configuration |
| `--output` | Parent directory for a new experiment |

`--dcp-mode learned` is the DCP training default and uses the normal 60 / 20 / 20 schedule on roads. `--dcp-mode classical` selects the optional frozen-dehazer segmentation workflow.

### Training outputs

```text
runs/paper/paired-road/coloraware/
├── config.json           # Full model, training and experiment configuration
├── split.json            # IDs, roots, grouping rule and clear-reference hashes
├── metrics.csv
├── final_metrics.json    # Validation metrics for the last model
├── dehaze/{best,last}.pth
├── seg/{best,last}.pth
├── joint/{best,last}.pth
├── best.pth              # Best checkpoint from the last executed stage
├── last.pth              # Final checkpoint from the last executed stage
└── plots/
```

Each stage starts from the previous stage's last parameters. Best checkpoints use PSNR for restoration, mIoU for segmentation, and `PSNR/30 + SSIM + mIoU + F1` for joint training; `--score-metric` changes the latter. Final metrics describe the last model, which may differ from the best checkpoint.

Choose a new `--output` for a new run. Existing training results are not overwritten, and the CLI currently has no interrupted-run resumption option.

## Prediction and evaluation

### Single image and batch prediction

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy --output results/batch
```

Joint prediction saves restored images, class-index PNG masks, overlays and comparison strips. Restoration-only weights produce segmentation when `--segmenter-checkpoint` is supplied. Neural inputs are internally padded to a multiple of 16; classical DCP uses the unpadded image. Outputs retain the original resolution. Optional `--resize H W` reduces inference cost, with outputs resized back afterwards.

### Evaluation

The evaluation follows the paper's three perspectives. These measurements describe different aspects of the output, so restoration scores and segmentation overlap should be considered together.

| Perspective | Metrics | Interpretation |
| :--- | :--- | :--- |
| Restoration quality | PSNR, SSIM ↑ | Pixel fidelity and structural similarity to the clear reference |
| Color preservation | ΔSat, CRerr ↓ | Differences in saturation and relative RGB channel proportions |
| Road segmentation | mIoU, mean Dice, F1 ↑ | Agreement between predicted regions and the annotated mask |

↑ Higher is better; ↓ lower is better. Segmentation metrics require ground-truth masks and are available for the paired road dataset.

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/road-eval
python -m dehaze_seg evaluate --checkpoint runs/sots/sots-indoor/coloraware/best.pth --dataset sots-indoor --data-root SOTS/indoor --split all --output results/sots-eval
```

Evaluation defaults to the saved validation split on a known training/validation root, and to `all` for independent benchmark roots. Road evaluation defaults to `val`. Use `--split-file` for a portable explicit split; with a shared segmenter its manifest supplies the default selection. All compared models use the same ordered sample list. A checkpoint trained with `--val-root` directs validation to that directory.

Known overlap with the training samples of either the dehazer or the shared segmenter is rejected, including copied clear files with matching hashes. `--allow-training-overlap` permits an explicitly labelled diagnostic and records overlap in the results. These checks cannot certify unknown training history or detect every transformed/re-encoded copy. Historical weights without a manifest reconstruct a requested split from seed/ratio and require unchanged data contents; they do not establish an independent test set.

`--metric-align crop` center-crops mismatched predictions and references to their common size; `resize` resizes the reference; `none` requires equal dimensions. Evaluation defaults to `--metric-resolution original`: even with `--resize`, outputs are upsampled before scoring. To match the road training validation grid, use the same checkpoint/split and **both** `--resize 512 512 --metric-resolution inference`. This scores predictions before output upsampling and resizes RGB references with PIL bilinear interpolation and masks with nearest-neighbor interpolation, as in the road loader:

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --resize 512 512 --metric-resolution inference --output results/road-eval-512
```

**Metric definitions:** `miou` and `mdice` are computed from the same `argmax` mask and confusion matrix, averaging background (class 0) and road (class 1). `iou_class_1` and `dice_class_1` are foreground-only scores. Hard `mdice` is different from `1 − soft Dice loss`. Classes absent from both masks contribute zero. With matching classes, masks and aggregation, `mdice >= miou`; mixing foreground Dice with macro IoU invalidates that comparison. Figure annotations must use the two corresponding columns from the same per-image row, without substituting dataset aggregates or loss values.

Per-image results (including per-class IoU/Dice) are saved in `metrics.csv`, aggregates in `summary.json`, and sample IDs, model configuration, checkpoint hash, references, metric definitions/resolution and shared-segmenter provenance in `protocol.json`. `segmentation_metrics.json` preserves per-image confusion matrices, mask hashes and actual scoring dimensions for independent checks. Restoration metrics are averaged per image; dataset segmentation scores are computed after summing confusion matrices, so they need not equal the mean of the CSV scores. Validation PSNR uses per-image averaging. `--save-images` exports original-resolution presentation images even when scoring at inference resolution; `--limit 1` processes one sample.

### Compare checkpoints

Compare dehazers under the shared frozen-segmenter protocol of Section 4.1.5:

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth runs/comparison/paired-road/c2pnet/best.pth --include-dcp --segmenter-checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --save-images --output results/comparison
```

The segmenter is loaded once, frozen and kept in evaluation mode. Its saved normalization is applied to every dehazer, including restoration-only weights and classical DCP. Its path and SHA-256 are saved with the results. Per-method segmenters are replaced. Omitting this option uses each checkpoint's own segmenter, which is a different protocol. Select the shared checkpoint before comparing methods; do not select it on test results. Historical shared weights accept `--segmenter-model`, `--segmenter-legacy-profile` and `--segmenter-model-config` when needed.

Classical DCP can also run on its own, using the same public model name:

```bash
python -m dehaze_seg predict --model dcp --input datasets/hazy/001.png --output results/dcp
python -m dehaze_seg evaluate --model dcp --segmenter-checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/dcp-eval
```

## Checkpoints

New checkpoints store complete `model_config`, training configuration and the actual data split, including roots and clear-reference hashes. The loader also supports historical `model_state`, `model_state_dict`, `state_dict` and `dehazer_state` containers and common wrapper prefixes. Recognized historical training `args` are translated into the corresponding configuration. Historical `gain_local` weights retain their original lower bound; corrected defaults apply to new experiments.

```bash
python -m dehaze_seg predict --checkpoint weights/colorawareunet.pth --input datasets/hazy/001.png --resize 512 512 --output results/legacy
python -m dehaze_seg predict --checkpoint weights/raw.pth --model coloraware --legacy-profile paper --input datasets/hazy/001.png --output results/raw
```

These paths refer to user-supplied local files, not bundled downloads. Metadata-free weights require an explicit configuration. `paper` selects the paper defaults; `legacy-infer` selects historical generic inference defaults, including ColorAwareUNet gain scale 0.65 without a gain floor. Other historical architectures can use `--model-config model.json`, with the same object structure as `config.json`'s `model_config` member. Parameter mismatches fail explicitly; partial loading is not used. Old script entrypoints and module paths have been removed.

## Ablations and visualization

### Registered ablations

```bash
python -m dehaze_seg ablate --list
```

<details>
<summary><strong>Run architecture, gain and attention ablations</strong></summary>

```bash
python -m dehaze_seg ablate --data-root datasets --experiments arch_baseline_unet,arch_no_color_gain,arch_no_refine,arch_full --output runs/ablation --device cuda --amp
python -m dehaze_seg ablate --data-root datasets --experiments gain_local,gain_amp,gain_no_min --output runs/gain-ablation --device cuda --amp
python -m dehaze_seg ablate --data-root datasets --experiments attention_none,attention_gate,attention_se,attention_both --output runs/attention-ablation --device cuda --amp
```

</details>

Additional registered experiments cover training strategy, loss, capacity, augmentation, normalization and segmentation directly from haze. `--experiments all` runs the full registry. Experiments share the trainer, split and metrics; outputs are grouped by experiment name with `ablation_summary.csv`.

### Reproduction notes

| Setting | Current implementation and scope |
| :--- | :--- |
| Table 4 local gain | `gain_local`: local tanh, scale 0.30, **no lower clamp** (`gain_min=None`) |
| Table 4 amplify-only gain | Keeps `1 + 0.30 × tanh(ReLU(raw))`; final gain weights start at zero and bias at **0.001**, allowing gradients through ReLU. The main tanh model retains zero initialization. |
| Historical amplify-only runs | All-zero initialization made the gain head inactive. Loading preserves the saved weights; correcting the experiment requires retraining. The manuscript's blanket zero-initialization statement needs this exception. |
| Weak / strong gain | Repository settings are `(scale, min)=(0.10, 0.98)` / `(0.50, 0.90)`; the manuscript does not specify these exact values. Local smoothing uses kernel 15. |
| Table 6 attention / SE | The four combinations are registered. The table's 5.878 / 5.944 / 6.097 / 6.164 M values match **joint-model** parameter counts, not the segmenter alone. The current runner uses the full staged schedule for each variant. |
| Ablation summary | `ablation_summary.csv` reports final-epoch validation metrics, not a re-evaluation of `best.pth`. Each stage continues from the preceding stage's last weights. |

The manuscript does not fully specify the attention experiment's training budget and frozen branches, or all model-selection settings. Confirm these against the original logs before claiming numerical reproduction. The trainer records active losses at each stage; it does not record the inactive restoration training loss during segmentation-only training, so its CSV alone cannot fully redraw Figure 6(a). CPU regression checks cover the corrected workflows; CUDA runs, full training and the paper's reported scores have not been revalidated.

### Visualization

```bash
python -m dehaze_seg visualize gain --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --samples 001 --output-dir results/color_gain
python -m dehaze_seg visualize components --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --sample 001 --output-dir results/components
python -m dehaze_seg visualize curves --metrics runs/paper/paired-road/coloraware/metrics.csv --output results/curves
python -m dehaze_seg visualize introduction --data-root datasets --sample 001 --visualization-dir results/color_gain --output-dir results/figures
```

Gain/component visualization requires a joint ColorAwareUNet checkpoint. The introduction panel consumes the matching sample produced by gain visualization. Each subcommand supports `--help`.

### Regenerate the segmentation comparison (Figure 7)

```bash
python -m dehaze_seg visualize segmentation --checkpoint weights/colorawareunet.pth --segmenter-checkpoint weights/colorawareunet.pth --include-dcp --data-root datasets --sample 003 --resize 512 512 --output results/figure7 --device cuda
```

This generates four columns: hazy input, **classical DCP**, ColorAwareUNet and ground truth. Both dehazers use the same frozen segmenter. Sample `003` is the first ID in the supplied historical checkpoint's validation split; it has no detected training-reference overlap. For other checkpoints, omit `--sample` to use their first validation ID, or select an ID from their saved split. Training samples and known reference overlap are rejected. This is a single validation example, not an independent test set or a reproduction of Table 1.

Scores are **mIoU / mDice**, computed from the same full 512×512 hard masks, including background and foreground. They are generated directly from the metric records. The red-box enlargements are illustrative and do not change the scoring region. Display panels preserve the original aspect ratio. `--roi X0 Y0 X1 Y1` sets one normalized crop for every panel. All prediction/target masks, confusion matrices, checkpoint/input hashes, CSV/JSON records and the caption accompany `figure7.png` (300 dpi) and `figure7.pdf`.

Supply additional actual experiment weights with `--checkpoint path1 path2 ...` to add other models. Missing model weights are never replaced with old figure labels or scores. `--include-dcp` uses the parameter-free classical implementation; a trained DCP extension must be supplied as a checkpoint and is labelled separately. Generated figures and local weights remain ignored by Git; the generation code is included in the repository.

## Project structure

```text
train.py                  # Direct training entrypoint
dehaze_seg/
├── models/               # Model definitions, registry and output adaptation
├── data/                 # Pairing, masks and synchronized augmentation
├── losses.py             # Restoration and segmentation objectives
├── metrics.py            # Image quality, color and segmentation metrics
├── engine/               # Training, inference, evaluation and checkpoints
├── experiments/          # Explicit ablation configurations and execution
├── visualization/        # Curves, gains, components and paper figures
└── cli.py                # Public command interface
tools/                    # Optional annotation and dataset utilities
tests/                    # CPU regression checks
```

The package provides `train`, `predict`, `evaluate`, `ablate` and `visualize` through `python -m dehaze_seg` or the installed `dehaze-seg` executable. See [dataset tools](tools/README.md) for optional utilities. Local datasets, weights, generated output, paper files and notebooks are ignored by Git; source and tests are tracked.

## Validation

```bash
python -m unittest discover -s tests -v
```

**22 CPU regression tests passed** in the local environment listed under [installation](#installation). They cover model forwards, learnable amplify-only gain, attention/SE variants, pairing, synchronized transforms, stage freezing, parameter updates, strict checkpoint round trips, scene isolation, shared frozen segmentation, classical DCP segmentation and learned DCP restoration/joint training. The joint DCP test also verifies that segmentation loss reaches the dehazing refinement head. Tests do not download datasets or VGG weights.

Additional local checks verified the 185 road triplets, a small three-stage training run, prediction/evaluation/ablation/visualization workflows, and inference with an existing historical checkpoint. **CUDA execution and full paper training were not validated.** No reproduction scores are claimed here.

## FAQ

<details>
<summary><strong>VGG download failure</strong></summary>

Cache `vgg16-397923af.pth` under `TORCH_HOME/hub/checkpoints/` (normally `~/.cache/torch/hub/checkpoints/` when unset). Explicit `--lam-perc 0` runs a different, perceptual-free objective.

</details>

<details>
<summary><strong>Slow CPU or insufficient GPU memory</strong></summary>

Reduce batch size or image size; tune CPU `--threads`. Changing training resolution changes the experiment.

</details>

<details>
<summary><strong>Missing data</strong></summary>

Check for an extra nested `datasets/` directory and matching stems.

</details>

<details>
<summary><strong>Checkpoint mismatch</strong></summary>

Supply its correct configuration or matching weights; do not hide architectural differences with non-strict loading.

</details>

## Citation

This repository accompanies the manuscript named at the top of this page. Refer to the final paper for experimental claims and evaluation protocols; regression tests are not a replacement for full reproduction. The following entry identifies the software repository. When the paper is published, also cite the bibliographic entry provided by the publisher.

```bibtex
@misc{colorawarenet_joint_model,
  title = {ColorAwareNet Joint Model: Color-Gain-Guided Image Dehazing and Semantic Segmentation},
  howpublished = {GitHub repository},
  url = {https://github.com/dongfangshiwen/colorawarenet-joint-model}
}
```

For questions or bug reports, open a [GitHub issue](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues) with the command, environment versions and error log.

---

<div align="center">

[Back to top](#colorawarenet) · [简体中文](README.zh-CN.md)

</div>
