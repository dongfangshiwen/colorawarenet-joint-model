# A Color-Gain-Guided, Color-Preserving Joint Framework for Image Dehazing and Semantic Segmentation

[中文](#中文) | [English](#english) | [Repository](https://github.com/dongfangshiwen/colorawarenet-joint-model)

**Training / 训练入口:** [`train.py`](train.py) · **Training implementation / 训练实现:** [`dehaze_seg/engine/train.py`](dehaze_seg/engine/train.py)

## 中文

### 简介

本仓库实现论文中的颜色增益引导去雾与语义分割框架：

```text
雾图 [0,1] → ColorAwareUNet → 去雾图 [0,1] → ImageNet 归一化 → LiteAttentionUNet → 分割标签
```

ColorAwareUNet 结合全局 RGB 增益、空间残差与细化分支；LiteAttentionUNet 使用深度可分离卷积，并提供 attention gate 和 SE 开关。训练包括去雾预训练、冻结去雾器的分割训练，以及联合微调。

这是代码仓库，不包含数据、训练权重、论文附件或 notebook。训练、推理和实验统一通过命令行运行。

### 安装

先克隆仓库并进入根目录，后续命令均在此目录运行：

```bash
git clone https://github.com/dongfangshiwen/colorawarenet-joint-model.git
cd colorawarenet-joint-model
```

需要 Python 3.10 或更高版本。建议在独立虚拟环境中安装彼此兼容的 PyTorch、torchvision，再安装本项目。

```bash
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；Linux 使用 `source .venv/bin/activate`。

已验证的 CPU 环境安装命令：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m dehaze_seg --help
```

使用 GPU 时，先安装适合本机驱动的 CUDA 版 PyTorch/torchvision，再执行 `python -m pip install -e .`。`--device auto` 自动选择 CUDA 或 CPU，`--device cuda` 明确要求 CUDA；`--amp` 仅在 CUDA 上启用混合精度。安装后也可使用 `dehaze-seg` 命令。

| 环境 | 说明 |
|---|---|
| 论文记录 | Ubuntu 22.04、Python 3.12、PyTorch 2.3.0、CUDA 12.1、32 GB vGPU |
| CPU 回归验证 | Windows、Python 3.12、PyTorch 2.8.0+cpu、torchvision 0.23.0 |

CPU 回归检查不代表完整论文结果复现；CUDA 训练需在相应硬件环境中另行验证。依赖范围用于安装解析，不表示所有版本组合均经过测试。

### 数据集

个人训练数据：**datasets.zip**，共 185 组配对道路场景样本。

- [百度网盘下载](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888)
- 提取码：**8888**

自行下载并解压，使目录结构如下；`--data-root` 指向包含这三个子目录的文件夹：

```text
datasets/
├── hazy/       # 001.png, 002.png, ...
├── clear/      # 同名清晰图
└── masks/      # 同名分割标注
```

按文件主名配对，扩展名可以不同。图像使用 RGB；二分类标注为背景 0、道路 1，也支持灰度 0/255。标注缩放使用最近邻插值，几何增强同步作用于图像和标注。

默认先排序样本 ID，再以 Python 随机种子 42 打乱，按 0.15 划分验证集：**训练 157 组，验证 28 组，无独立测试集**。每次训练保存 `split.json`，后续道路评估优先使用该文件。

其他数据集需自行准备，不包含在上述道路数据分享中：

| `--dataset` | 目录与配对规则 |
|---|---|
| `sots-indoor` / `sots-outdoor` | `hazy/` 和 `clear/` 或 `gt/`；先匹配完整主名，再将 `1400_1` 匹配到 `1400` |
| `hsts` | `synthetic/synthetic/` 为雾图，`synthetic/original/` 为同名清晰图 |
| `nh-haze` | `hazy/01_hazy.png` 与 `clear/01_GT.png`，也支持同名配对 |

这些入口默认只训练去雾，不生成分割标签。HSTS 保留同步裁剪、翻转和旋转增强，`--train-repeats` 控制每轮重复次数；增强 HSTS 衍生数据实验不能作为未修改官方 HSTS 协议的结果。HSTS 的无参考真实图像可直接通过 `predict --input` 推理，不计算全参考指标。

### 模型与论文默认参数

| 名称 | 内容 |
|---|---|
| `coloraware` | 论文主去雾模型 ColorAwareUNet |
| `c2pnet` | 仓库中的 C2PNet 实现 |
| `dcp` | DCP 加可训练细化分支；当前默认会启用细化，不能视为纯经典 DCP |
| `ffanet` | 仓库中的 FFA-Net 实现 |
| `grid` | 仓库中的 GridDehazeNet 实现 |
| `psd` | 仓库中的 PSDDehazeNet 实现 |

对比模型保留本仓库已有实现与参数，不宣称与原作者官方代码、权重或结果完全一致。

| 参数 | 默认值 |
|---|---|
| 去雾基础通道 | 32 |
| 增益 | global、tanh、scale=0.30、min=0.95，有效范围 [0.95, 1.30] |
| 残差 / 细化系数 | 0.50 / 0.25 |
| 分割 | 2 类、基础通道 32、width multiplier=1.0 |
| Attention / SE / 辅助头 | 开 / 关 / 关 |
| ImageNet 归一化 | 开 |
| 输入尺寸 / batch size | 512×512 / 2 |
| 优化器 | AdamW，lr=1e-4，weight decay=1e-5 |
| 三阶段轮数 | 60 / 20 / 20 |

去雾损失为 `L1 + 0.40 × SSIM_loss + 0.05 × VGG_perceptual`；SSIM 使用 11×11 均值窗口，损失为 `(1 − SSIM) / 2`。感知损失使用冻结的 ImageNet VGG16 第 3、8、15 层特征。分割损失为 CE＋类别平均 soft Dice，联合阶段两项任务权重均为 1。

MSE、梯度差、ΔSat 和 CRerr 不参与默认优化；侧输出只用于兼容和可视化。局部增益与只放大增益属于消融设置。全局增益贡献热图表示 `mean(abs(input × (gain−1)))`，不是空间增益图。

### 训练

训练入口是根目录的 [`train.py`](train.py)。它调用 [`dehaze_seg/engine/train.py`](dehaze_seg/engine/train.py) 中的公共实现，参数与 `python -m dehaze_seg train` 相同；不需要再写 `train` 子命令。

1. 按上文安装依赖，把数据解压到 `datasets/hazy`、`datasets/clear`、`datasets/masks`。
2. 运行 `python train.py --help` 查看参数。
3. 运行下方训练命令；默认依次执行 60 轮去雾、20 轮分割、20 轮联合微调。
4. 使用生成的 `best.pth` 执行下方推理与评估命令。

道路数据三阶段训练：

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --amp
```

完整参数示例（单行命令可用于 PowerShell 或 Linux shell）：

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --pretrain-dehaze-epochs 60 --pretrain-seg-epochs 20 --finetune-epochs 20 --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/paper-explicit --amp
```

只检查环境与训练流程，可在 CPU 上执行以下缩小规模的实验。它会读取数据并生成独立结果，不用于复现论文指标；显式关闭感知损失以避免下载 VGG 权重：

```bash
python train.py --data-root datasets --device cpu --resize 64 64 --batch-size 2 --pretrain-dehaze-epochs 1 --pretrain-seg-epochs 1 --finetune-epochs 1 --lam-perc 0 --output runs/smoke --no-plots
```

训练对比模型，仅改变模型名称：

```bash
python train.py --dataset paired-road --data-root datasets --model c2pnet --output runs/comparison --amp
```

SOTS 去雾实验示例使用 80 轮、batch size 4、lr=5e-4：

```bash
python train.py --dataset sots-indoor --data-root SOTS/indoor --model coloraware --pretrain-dehaze-epochs 80 --batch-size 4 --lr 5e-4 --output runs/sots --amp
python train.py --dataset sots-outdoor --data-root SOTS/outdoor --model coloraware --pretrain-dehaze-epochs 80 --batch-size 4 --lr 5e-4 --output runs/sots --amp
python train.py --dataset hsts --data-root HSTS --train-repeats 4 --output runs/hsts --amp
python train.py --dataset nh-haze --data-root NH-HAZE --output runs/nh-haze --amp
```

基准数据集只使用 `--pretrain-dehaze-epochs`，不会执行分割和联合阶段。通用新实验默认学习率仍为 1e-4，示例中的 SOTS 学习率是显式实验设置。

常用选项：`--pretrain-dehaze-epochs`、`--pretrain-seg-epochs`、`--finetune-epochs`、`--batch-size`、`--resize H W`、`--workers`、`--use-se`、`--no-attention`、`--no-imagenet-norm`。默认 `--workers 0` 便于跨平台启动；可按机器增加。

输出示例：

```text
runs/paper/paired-road/coloraware/
├── config.json          # 完整模型、训练和实验配置
├── split.json           # 实际训练/验证样本 ID
├── metrics.csv
├── final_metrics.json   # 最后一轮模型的验证指标
├── dehaze/{best,last}.pth
├── seg/{best,last}.pth
├── joint/{best,last}.pth
├── best.pth             # 最后一个已执行阶段的最优模型
├── last.pth             # 最后一个已执行阶段的末轮模型
└── plots/
```

阶段按前一阶段的末轮参数继续训练。去雾阶段以 PSNR、分割阶段以 mIoU 选择最优权重；联合阶段默认使用 `PSNR/30 + SSIM + mIoU + F1`，可用 `--score-metric` 修改。输出目录已有训练结果时需使用新的 `--output`，避免覆盖。当前 CLI 不提供断点续训。

### 推理与评估

单图预测与目录预测：

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy --output results/batch
```

输出包括去雾图、类别编号 PNG、分割叠加图和拼图；纯去雾权重只输出去雾结果。预测保留原图尺寸，内部补齐至 16 的倍数；可用 `--resize H W` 降低推理开销，结果会插值回原尺寸。

道路验证集与 SOTS 评估：

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/road-eval
python -m dehaze_seg evaluate --checkpoint runs/sots/sots-indoor/coloraware/best.pth --dataset sots-indoor --data-root SOTS/indoor --split all --output results/sots-eval
```

道路评估默认 `val`，其他数据集默认 `all`。可通过 `--split-file` 指定保存的划分。旧权重无划分文件时，按保存的 seed/val_ratio 重建；重建要求数据集合保持一致。训练时对 SOTS 的内部划分用于监控，按雾图 ID 划分而非按场景分组，不能将其称为独立标准测试结果。

`--metric-align crop` 默认将尺寸不一致的预测与参考图中心裁剪到共同区域；`resize` 将参考图缩放到预测尺寸；`none` 要求尺寸完全相同。训练保持原有数据预处理方式，评估默认原尺寸，因此训练日志与原尺寸评估结果可能不同。

评估保存逐图 `metrics.csv` 和汇总 `summary.json`。恢复指标按图平均，分割指标由全数据混淆矩阵计算。与旧脚本相比，验证 PSNR 改为逐图平均，不再先按 batch 混合 MSE，以避免 batch size 改变汇总值。`--save-images` 保存评估图片，`--limit 1` 可只检查一张。

多个权重可在同一命令中比较，生成各自结果、汇总表和对比拼图：

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth runs/comparison/paired-road/c2pnet/best.pth --input datasets/hazy --limit 5 --output results/comparison
```

### 旧权重

新权重保存完整 `model_config`，加载时自动恢复结构和数值配置。支持历史 `model_state`、`model_state_dict`、`state_dict`、`dehazer_state` 字典，以及常见外层前缀。历史联合训练权重中保存的 `args` 会映射为论文训练配置。

```bash
python -m dehaze_seg predict --checkpoint weights/colorawareunet.pth --input datasets/hazy/001.png --resize 512 512 --output results/legacy
```

上述权重路径仅表示已有的本地文件，仓库不附带权重下载。无配置的裸参数文件需要明确选择历史设置：

```bash
python -m dehaze_seg predict --checkpoint weights/raw.pth --model coloraware --legacy-profile paper --input datasets/hazy/001.png --output results/raw
```

`paper` 使用论文默认配置，`legacy-infer` 对应旧通用推理入口的默认配置（例如 ColorAwareUNet gain=0.65、无 gain 下限）。对于其他历史结构，用 `--model-config model.json` 提供完整模型配置；格式与训练 `config.json` 中的 `model_config` 对象一致。结构不匹配会明确报错，不会部分加载后继续推理。旧模块路径和旧命令入口不再保留。

### 消融与可视化

```bash
python -m dehaze_seg ablate --list
python -m dehaze_seg ablate --data-root datasets --experiments arch_baseline_unet,arch_no_color_gain,arch_no_refine,arch_full --output runs/ablation --amp
python -m dehaze_seg ablate --data-root datasets --experiments gain_local,gain_amp,gain_no_min --output runs/gain-ablation --amp
python -m dehaze_seg ablate --data-root datasets --experiments attention_none,attention_gate,attention_se,attention_both --output runs/attention-ablation --amp
```

还提供训练策略、损失、模型容量、数据增强、输入归一化和直接雾图分割实验。`--experiments all` 运行完整注册表。所有消融共享训练流程、随机划分和指标；结果按实验名称保存，并汇总到 `ablation_summary.csv`。

```bash
python -m dehaze_seg visualize gain --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --samples 001 --output-dir results/color_gain
python -m dehaze_seg visualize components --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --sample 001 --output-dir results/components
python -m dehaze_seg visualize curves --metrics runs/paper/paired-road/coloraware/metrics.csv --output results/curves
python -m dehaze_seg visualize introduction --data-root datasets --sample 001 --visualization-dir results/color_gain --output-dir results/figures
```

增益和组件可视化要求联合 ColorAwareUNet 权重。Introduction 拼图使用前一步生成的相同 sample 结果。各子命令可通过 `--help` 查看选项。

### 代码结构与检查

```text
train.py             # 直接训练入口
dehaze_seg/
├── models/          # 论文模型、对比模型、输出适配与统一注册表
├── data/            # 道路三元组和各基准配对/增强
├── losses.py        # 去雾与分割目标
├── metrics.py       # 图像质量、颜色与分割指标
├── engine/          # train.py 训练实现、评估、推理和权重加载
├── experiments/     # 显式消融配置与执行
├── visualization/   # 曲线、增益、组件及论文拼图
└── cli.py           # 公开命令入口
tools/               # 可选标注、合成雾与文件整理工具
tests/               # CPU 回归检查
```

```bash
python -m unittest discover -s tests -v
```

测试不下载数据或 VGG 权重，覆盖模型前向、增益/attention/SE、数据配对、同步增强、阶段冻结、参数更新和权重保存/加载。

常见问题：

- **VGG 下载失败**：默认感知损失需要 `vgg16-397923af.pth`，放在 `TORCH_HOME/hub/checkpoints/` 缓存中；未设置 `TORCH_HOME` 时通常为 `~/.cache/torch/hub/checkpoints/`。也可显式 `--lam-perc 0` 做无感知损失实验，不能将其当作论文完整目标。
- **CPU 很慢或显存不足**：缩小 `--batch-size`、`--resize`；CPU 可调整 `--threads`。缩小训练尺寸会改变实验设置。
- **找不到数据**：检查解压后是否多嵌套了一层 `datasets/`，并核对文件主名和目录名。
- **模型不匹配**：提供正确旧配置或对应权重，不要通过非严格加载掩盖结构差异。

可选数据工具见 [tools/README.md](tools/README.md)。`.gitignore` 排除数据、权重、输出、论文文件和 notebook；源码和测试正常跟踪。

### 论文与引用

本仓库对应页首同名论文的方法实现。论文指标应依据最终论文中的实验协议与结果报告；这里的回归测试不替代完整复现实验。若引用代码，可使用文末的软件仓库条目；正式论文发表后，应同时引用出版社提供的论文条目。

问题反馈请通过 [GitHub Issues](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues) 提交，并附上运行命令、环境版本与错误日志。

---

## English

### Overview

This repository implements the color-gain-guided joint dehazing and semantic segmentation framework described in the paper named above. ColorAwareUNet combines global RGB gain, spatial residual prediction and refinement. The restored image is ImageNet-normalized and passed to LiteAttentionUNet, which uses depthwise-separable convolutions with configurable attention gates and SE blocks.

Training consists of restoration pretraining, segmentation training with a frozen dehazer, and joint fine-tuning. This is a code-only release: datasets, checkpoints, paper attachments and notebooks are excluded. Training starts from `train.py`; prediction, evaluation, ablations and visualization use the package CLI.

### Installation and environment

Clone the repository and run subsequent commands from its root:

```bash
git clone https://github.com/dongfangshiwen/colorawarenet-joint-model.git
cd colorawarenet-joint-model
```

Use Python 3.10+ and create a virtual environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux **before installing dependencies**. The verified CPU package combination is:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m dehaze_seg --help
```

For GPU execution, install a compatible CUDA build of PyTorch and torchvision before installing this project. `--device auto` selects CUDA when available; `--device cuda` requires it. `--amp` enables mixed precision on CUDA only. Installation also provides the `dehaze-seg` executable.

The paper reports Ubuntu 22.04, Python 3.12, PyTorch 2.3.0, CUDA 12.1 and a 32 GB vGPU. CPU regression checks were run on Windows with Python 3.12, PyTorch 2.8.0+cpu and torchvision 0.23.0. CUDA training and complete experimental reproduction were not performed. Dependency ranges do not imply that every version combination was tested.

### Datasets

The paired road dataset contains **185 hazy/clear/mask triplets**:

- Archive: **datasets.zip**
- [Baidu Netdisk download](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888)
- Extraction code: **8888**

Download and extract manually. `--data-root` must contain `hazy/`, `clear/` and `masks/`, with matching file stems such as `001.png`. File extensions may differ. Images are RGB; binary masks use background=0 and road=1, with 0/255 grayscale masks also supported. Masks use nearest-neighbor resizing and share geometric transforms with their images.

Sorted sample IDs are shuffled using Python random seed 42. A validation ratio of 0.15 produces **157 training and 28 validation samples**, with **no independent test split**. Training saves the exact IDs in `split.json`.

Other datasets must be prepared separately:

| Dataset option | Required layout and pairing |
|---|---|
| `sots-indoor`, `sots-outdoor` | `hazy/` and `clear/` or `gt/`; exact stem first, then scene prefix (`1400_1` → `1400`) |
| `hsts` | `synthetic/synthetic/` for haze, `synthetic/original/` for matching clear images |
| `nh-haze` | `hazy/01_hazy.png` and `clear/01_GT.png`; same-stem pairs are also supported |

These benchmarks use restoration-only supervision. No dummy segmentation masks are generated. HSTS keeps paired crop, flip and rotation augmentation; `--train-repeats` controls repeated training samples. Results on augmented HSTS-derived data are not results under the unmodified official HSTS protocol. Unpaired real images can be processed with `predict --input`, without full-reference evaluation.

### Models and defaults

Supported dehazers are `coloraware`, `c2pnet`, `dcp`, `ffanet`, `grid` and `psd`. All road experiments use LiteAttentionUNet for segmentation. Baselines retain this repository's implementations and are not claimed to reproduce the original authors' official code or scores. In particular, the DCP implementation enables a trainable refinement branch by default and is not a pure classical DCP baseline.

The main configuration uses ColorAwareUNet width 32, global tanh gain with scale 0.30 and minimum 0.95, residual scale 0.50 and refinement scale 0.25. The effective gain range is [0.95, 1.30]. Segmentation uses two classes, width 32, width multiplier 1.0, attention on, SE off, auxiliary head off and ImageNet normalization on.

Default training uses 512×512 images, batch size 2, AdamW with learning rate 1e-4 and weight decay 1e-5, and 60/20/20 epochs. The restoration loss is `L1 + 0.40 × SSIM_loss + 0.05 × VGG_perceptual`, where `SSIM_loss=(1−SSIM)/2` uses an 11×11 averaging window. Perceptual loss uses frozen ImageNet VGG16 feature indices 3, 8 and 15. Segmentation uses CE plus class-averaged soft Dice. Both task weights default to 1 during joint fine-tuning.

MSE, gradient discrepancy, saturation deviation and chromaticity ratio error are monitoring metrics, not additional default objectives. Side outputs are retained for visualization and compatibility without auxiliary supervision. Local and amplify-only gains are ablations. A global gain contribution heatmap shows `mean(abs(input × (gain−1)))`, not a spatial gain field.

### Training

Use the root [`train.py`](train.py) entrypoint. Its implementation is [`dehaze_seg/engine/train.py`](dehaze_seg/engine/train.py). It accepts the same options as `python -m dehaze_seg train`; do not add another `train` subcommand.

1. Install dependencies and prepare `datasets/hazy`, `datasets/clear` and `datasets/masks`.
2. Run `python train.py --help` to inspect the available options.
3. Start training below. The default schedule runs 60 restoration, 20 segmentation and 20 joint epochs.
4. Use the generated `best.pth` for prediction or evaluation.

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --amp
python train.py --dataset paired-road --data-root datasets --model c2pnet --output runs/comparison --amp
python train.py --dataset sots-indoor --data-root SOTS/indoor --model coloraware --pretrain-dehaze-epochs 80 --batch-size 4 --lr 5e-4 --output runs/sots --amp
python train.py --dataset sots-outdoor --data-root SOTS/outdoor --model coloraware --pretrain-dehaze-epochs 80 --batch-size 4 --lr 5e-4 --output runs/sots --amp
python train.py --dataset hsts --data-root HSTS --train-repeats 4 --output runs/hsts --amp
python train.py --dataset nh-haze --data-root NH-HAZE --output runs/nh-haze --amp
```

The SOTS examples explicitly use the experiment settings of 80 epochs, batch size 4 and lr=5e-4; the shared default remains lr=1e-4. Benchmark training executes only `--pretrain-dehaze-epochs`.

An explicit paper configuration and a short CPU workflow check are shown below. These single-line commands work in both PowerShell and Linux shells. The CPU check uses smaller images and disables perceptual loss to avoid downloading VGG weights; it is not an experiment for reproducing paper scores.

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --pretrain-dehaze-epochs 60 --pretrain-seg-epochs 20 --finetune-epochs 20 --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/paper-explicit --amp
python train.py --data-root datasets --device cpu --resize 64 64 --batch-size 2 --pretrain-dehaze-epochs 1 --pretrain-seg-epochs 1 --finetune-epochs 1 --lam-perc 0 --output runs/smoke --no-plots
```

Important options include `--pretrain-seg-epochs`, `--finetune-epochs`, `--batch-size`, `--resize H W`, `--workers`, `--use-se`, `--no-attention` and `--no-imagenet-norm`. Workers default to 0 for portable startup. Use `--help` for the full list.

Outputs live under `<output>/<dataset>/<model>/`, containing `config.json`, `split.json`, `metrics.csv`, `final_metrics.json`, stage-specific `best.pth`/`last.pth`, root `best.pth`/`last.pth`, and `plots/`. Root checkpoints belong to the last executed stage. Final metrics describe the last model, not necessarily the best checkpoint.

Each stage starts from the previous stage's last parameters. Best checkpoints use PSNR for restoration, mIoU for segmentation, and `PSNR/30 + SSIM + mIoU + F1` for joint training; `--score-metric` changes the latter. Choose a new output location for a new run; existing training output is not overwritten. The CLI currently does not implement interrupted-run resumption.

### Prediction and evaluation

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy --output results/batch
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/road-eval
python -m dehaze_seg evaluate --checkpoint runs/sots/sots-indoor/coloraware/best.pth --dataset sots-indoor --data-root SOTS/indoor --split all --output results/sots-eval
```

Joint prediction saves restored images, class-index PNG masks, overlays and comparison strips. Restoration-only weights do not produce segmentation. Images are internally padded to a multiple of 16; outputs retain the original resolution. Optional `--resize H W` reduces inference cost, with outputs resized back afterwards.

Evaluation defaults to `val` for paired roads and `all` for benchmarks. Use `--split-file` to select a saved split. When a historical checkpoint lacks a manifest, the split is reconstructed from its seed and ratio; this requires the same dataset contents. Internal SOTS splits use hazy-image IDs rather than scene groups and must not be described as independent standard test results.

`--metric-align crop` center-crops mismatched predictions and references to their common size; `resize` resizes the reference; `none` requires equal dimensions. Training uses its configured resize, while evaluation defaults to native resolution, so their results can differ.

Per-image results are saved in `metrics.csv`, with aggregate results in `summary.json`. Restoration metrics are averaged per image; segmentation metrics use the full confusion matrix. Validation PSNR now uses per-image averaging rather than the historical batch-aggregated MSE, making the aggregate independent of validation batch size. `--save-images` exports evaluation images; `--limit 1` processes one sample.

Compare multiple checkpoints in one run:

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth runs/comparison/paired-road/c2pnet/best.pth --input datasets/hazy --limit 5 --output results/comparison
```

This writes separate model outputs, a summary comparison table and image grids.

### Historical checkpoints

New checkpoints store complete `model_config` metadata. The loader also supports historical `model_state`, `model_state_dict`, `state_dict` and `dehazer_state` containers and common wrapper prefixes. Recognized historical training `args` are translated into the corresponding configuration.

```bash
python -m dehaze_seg predict --checkpoint weights/colorawareunet.pth --input datasets/hazy/001.png --resize 512 512 --output results/legacy
python -m dehaze_seg predict --checkpoint weights/raw.pth --model coloraware --legacy-profile paper --input datasets/hazy/001.png --output results/raw
```

These paths refer to user-supplied local files, not bundled downloads. Metadata-free weights require an explicit configuration. `paper` selects the paper defaults; `legacy-infer` selects historical generic inference defaults, including ColorAwareUNet gain scale 0.65 without a gain floor. Other historical architectures can use `--model-config model.json`, with the same object structure as `config.json`'s `model_config` member. Parameter mismatches fail explicitly; partial loading is not used. Old script entrypoints and module paths have been removed.

### Ablations and visualization

```bash
python -m dehaze_seg ablate --list
python -m dehaze_seg ablate --data-root datasets --experiments arch_baseline_unet,arch_no_color_gain,arch_no_refine,arch_full --output runs/ablation --amp
python -m dehaze_seg ablate --data-root datasets --experiments gain_local,gain_amp,gain_no_min --output runs/gain-ablation --amp
python -m dehaze_seg ablate --data-root datasets --experiments attention_none,attention_gate,attention_se,attention_both --output runs/attention-ablation --amp
```

Additional registered experiments cover training strategy, loss, capacity, augmentation, normalization and segmentation directly from haze. `--experiments all` runs the full registry. Experiments share the trainer, split and metrics; outputs are grouped by experiment name with `ablation_summary.csv`.

```bash
python -m dehaze_seg visualize gain --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --samples 001 --output-dir results/color_gain
python -m dehaze_seg visualize components --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --sample 001 --output-dir results/components
python -m dehaze_seg visualize curves --metrics runs/paper/paired-road/coloraware/metrics.csv --output results/curves
python -m dehaze_seg visualize introduction --data-root datasets --sample 001 --visualization-dir results/color_gain --output-dir results/figures
```

Gain/component visualization requires a joint ColorAwareUNet checkpoint. The introduction panel consumes the matching sample produced by gain visualization. Each subcommand supports `--help`.

### Layout, checks and troubleshooting

`train.py` is the direct training entrypoint and `dehaze_seg/engine/train.py` contains the training loop. `dehaze_seg/models` contains model definitions and the registry; `data` handles pairing and transforms; `losses.py` and `metrics.py` implement objectives and measurements; `engine` owns training, inference and checkpoint loading; `experiments` holds explicit ablation configurations; `visualization` produces figures. `tools` holds optional dataset utilities and `tests` holds CPU regression checks.

```bash
python -m unittest discover -s tests -v
```

Tests do not download datasets or VGG weights. They check forwards, gain/attention/SE variants, pairing, synchronized transforms, stage freezing, parameter updates and checkpoint round trips.

- **VGG download failure:** cache `vgg16-397923af.pth` under `TORCH_HOME/hub/checkpoints/` (normally `~/.cache/torch/hub/checkpoints/` when unset). Explicit `--lam-perc 0` runs a different, perceptual-free objective.
- **Slow CPU or insufficient GPU memory:** reduce batch size or image size; tune CPU `--threads`. Changing training resolution changes the experiment.
- **Missing data:** check for an extra nested `datasets/` directory and matching stems.
- **Checkpoint mismatch:** supply its correct configuration or matching weights; do not hide architectural differences with non-strict loading.

See [tools/README.md](tools/README.md) for optional annotation, fog synthesis and filename utilities. Git ignores local datasets, weights, generated output, paper files and notebooks while tracking source and tests.


### Paper and citation

This repository accompanies the manuscript named at the top of this page. Refer to the final paper for experimental claims and evaluation protocols; regression tests are not a replacement for full reproduction. The following entry identifies the software repository. When the paper is published, also cite the bibliographic entry provided by the publisher.

```bibtex
@misc{colorawarenet_joint_model,
  title = {ColorAwareNet Joint Model: Color-Gain-Guided Image Dehazing and Semantic Segmentation},
  howpublished = {GitHub repository},
  url = {https://github.com/dongfangshiwen/colorawarenet-joint-model}
}
```

For questions or bug reports, open a [GitHub issue](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues) with the command, environment versions and error log.
