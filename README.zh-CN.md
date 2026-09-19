<div align="center">

# ColorAwareNet

**颜色增益引导的图像去雾与语义分割**

[English](README.md) · **简体中文**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.0_GPU-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](#installation)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](#installation)
[![Training](https://img.shields.io/badge/Entry-train.py-0F766E?style=flat-square)](train.py)

[训练入口](train.py) · [数据下载](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888) · [GitHub Issues](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues)

</div>

> **论文** · *A Color-Gain-Guided, Color-Preserving Joint Framework for Image Dehazing and Semantic Segmentation*

**快速导航：** [项目介绍](#overview) · [开始使用](#quick-start) · [安装](#installation) · [数据集](#datasets) · [方法](#method) · [训练](#training) · [评估](#prediction-and-evaluation) · [权重](#checkpoints) · [实验](#ablations-and-visualization) · [常见问题](#faq) · [引用](#citation)

<a id="overview"></a>

## 项目介绍

**ColorAwareNet 将图像去雾与道路分割连接为一个可联合训练的流程。** 本仓库提供页首论文的 PyTorch 实现：输入一张雾天 RGB 图像，先由 **ColorAwareUNet** 恢复图像，再由 **LiteAttentionUNet** 根据去雾结果预测道路与背景的二分类标注。

论文关注的问题是：如何在改善可见度、保持颜色的同时，保留有助于语义分割的场景结构。雾会降低对比度、模糊目标边界，而去雾产生的伪影也可能改变分割网络识别道路时依赖的线索。因此，框架将图像恢复与语义分割纳入同一训练流程，并从图像质量、颜色一致性和分割准确性三个方面进行评估。

[![论文图 2：联合框架的总体网络架构、训练流程与推理过程。](docs/assets/framework.png)](docs/assets/framework.png)

*论文图 2：联合框架的总体架构、训练流程与推理过程。点击图片可查看原始分辨率。*

### 核心思路

- **显式的 RGB 颜色增益。** 去雾网络为每张图像预测三个增益值，分别作用于红、绿、蓝通道。同一通道的增益在整幅图像上共享，便于直接观察模型如何调整整体颜色。
- **局部细节恢复。** 空间残差与细化分支补充全局增益，使模型能在不同区域施加不同修正，恢复道路边界、标线等局部结构。
- **由分割目标参与优化的去雾。** 轻量注意力分割网络从去雾图中学习语义。在分别预训练后，联合微调使分割损失也能更新去雾网络，与图像恢复目标共同参与优化。

推理时**只需输入雾图**。清晰参考图和道路标注用于训练监督与配对评估。

### 本仓库可以完成什么

| 工作流程 | 数据 / 配置 | 主要输出 |
| :--- | :--- | :--- |
| [训练联合模型](#training) | 对齐的道路雾图、清晰图与标注三元组 | 三阶段权重、实际数据划分与训练曲线 |
| [开展去雾实验](#datasets) | SOTS indoor/outdoor、增强 HSTS，以及五种保留的对比模型 | 去雾图与图像质量、颜色指标 |
| [预测与评估](#prediction-and-evaluation) | 单张图像、图像目录或配对验证集 | 去雾 RGB 图、道路标注、叠加图与指标汇总 |
| [分析模型组件](#ablations-and-visualization) | 结构、增益、注意力和训练策略等注册式消融 | 实验汇总、增益可视化与组件图 |

道路实验使用 **185 组配对样本**，划分为 **157 组训练 / 28 组验证**，无独立测试集。SOTS 和增强 HSTS 用于去雾评估，不提供语义标注。配对规则与结果报告协议见[数据集](#datasets)和[评估说明](#prediction-and-evaluation)。

本仓库公开源码、使用文档与论文中的网络架构图。数据和权重需在本地准备；论文全文与 notebook 不包含在代码发布中。

---

<a id="quick-start"></a>

## 快速开始

**训练入口是根目录的 [`train.py`](train.py)**，公共训练循环位于 [`dehaze_seg/engine/train.py`](dehaze_seg/engine/train.py)。

1. 完成[安装](#installation)，后续命令均在仓库根目录运行。
2. 下载 [datasets.zip](https://pan.baidu.com/s/18BtKG8-QHzRhfCGqjiuoUA?pwd=8888)，提取码 **8888**，按[数据目录说明](#datasets)解压。
3. 启动论文默认的 60 / 20 / 20 三阶段训练：

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --device cuda --amp
```

训练完成后，用生成的权重预测一张图像：

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
```

将 `001.png` 替换为实际图像文件名。训练示例通过 `--device cuda --amp` 明确使用 CUDA GPU 和混合精度；预测默认自动选择可用 GPU。较小规模的 CPU 流程检查见[训练说明](#training)；默认感知损失需要预训练 VGG16 权重，离线缓存方式见[常见问题](#faq)。

<a id="installation"></a>

## 安装

先克隆仓库并进入根目录，后续命令均在此目录运行：

```bash
git clone https://github.com/dongfangshiwen/colorawarenet-joint-model.git
cd colorawarenet-joint-model
```

建议使用与论文一致的 **Python 3.12**（项目最低要求为 Python 3.10），并创建独立虚拟环境。

```bash
python -m venv .venv
```

激活环境：Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`；Linux 使用 `source .venv/bin/activate`。

### GPU 环境安装

论文第 4.1.3 节记录的环境为 **PyTorch 2.3.0、CUDA 12.1**。在具备 NVIDIA GPU 和兼容驱动的机器上，按 [PyTorch 官方版本表](https://pytorch.org/get-started/previous-versions/#v230)安装 CUDA 版 PyTorch 及对应的 torchvision：

```bash
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e . "numpy<2"
python -m dehaze_seg --help
```

这里将 NumPy 限制在 1.x，以避免旧版 PyTorch 环境中的二进制兼容问题；这是安装兼容约束，不是论文中记录的 NumPy 版本。说明见 [NumPy 官方兼容性指南](https://numpy.org/doc/stable/user/troubleshooting-importerror.html#downstream-importerror-attributeerror-or-c-api-abi-incompatibility)。

训练前检查 GPU 是否可用：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
```

`--device cuda` 明确要求使用可见的 CUDA GPU，`--amp` 启用 GPU 混合精度。CLI 默认的 `--device auto` 在 CUDA 可用时自动选择 GPU。安装后也可使用 `dehaze-seg` 命令。

<details>
<summary><strong>用于本地流程检查的 CPU 安装方式</strong></summary>

在另一个独立虚拟环境中，可安装本机已验证的 CPU 组合：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

运行[训练说明](#training)中的缩小规模示例，使用 `--device cpu`。这是本地验证环境，论文实验使用 GPU。

</details>

<details>
<summary><strong>论文环境与本机验证环境</strong></summary>

| 环境 | 说明 |
|---|---|
| 论文软件环境 | Ubuntu 22.04、Python 3.12、PyTorch 2.3.0、CUDA 12.1 |
| 论文云端硬件 | 1 张 32 GB vGPU、16 个 Intel Xeon Platinum 8352V vCPU、62 GB RAM |
| CPU 回归验证 | Windows、Python 3.12、PyTorch 2.8.0+cpu、torchvision 0.23.0 |

上述软件配置来自论文记录。本机 CPU 用于回归检查；后续已在 RTX 3080 Ti、PyTorch 2.3.0+cu121 上核验 CUDA 推理，未重新执行完整训练。依赖范围用于安装解析，不表示所有版本组合均经过测试。

</details>

<a id="datasets"></a>

## 数据集

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

新道路实验默认先按文件内容完全相同的清晰参考图分组，再以 seed=42、验证比例 0.15 划分。本地核查的 185 组数据仍得到**训练 157 组、验证 28 组，无独立测试集**。整个场景组不会拆开；其他数据的实际数量可能因组大小略有变化。每次训练在 `split.json` 和权重中保存实际 ID、分组规则与哈希。

本地数据有五组共用清晰参考图的样本：`058/114`、`062/130`、`063/140`、`177/179`、`181/182`。历史按 ID 划分时，验证样本 `114`、`181` 与训练集跨组共用参考图。订正的场景划分改变了具体样本归属，需要重新训练，不能宣称等同于原实验划分或成绩。`--split-unit sample` 可用于核查历史 ID 划分，并记录参考图重叠。旧权重沿用其保存的划分；不能直接换成新划分评估旧权重，就认为数据已独立。

### 基准数据集

其他数据集需自行准备，不包含在上述道路数据分享中：

| `--dataset` | 目录与配对规则 |
|---|---|
| `sots-indoor` / `sots-outdoor` | `hazy/` 和 `clear/` 或 `gt/`；先匹配完整主名，再将 `1400_1` 匹配到 `1400` |
| `hsts` | `synthetic/synthetic/` 为雾图，`synthetic/original/` 为同名清晰图 |

这些入口默认只训练去雾，不生成分割标签。HSTS 保留同步裁剪、翻转和旋转增强，`--train-repeats` 控制每轮重复次数；增强 HSTS 衍生数据实验不能作为未修改官方 HSTS 协议的结果。HSTS 的无参考真实图像可直接通过 `predict --input` 推理，不计算全参考指标。

<a id="method"></a>

## 方法与模型

### 从雾图到道路标注

**1. 恢复颜色与空间细节。** ColorAwareUNet 采用带跳跃连接的 U-Net 编码器—解码器。瓶颈特征用于预测全局 RGB 增益，解码器输出与图像同尺寸的残差；细化分支结合输入图、粗恢复图、残差和解码特征，生成最终修正。论文主配置对应：

```text
颜色增益 = max(0.95, 1 + 0.30 × tanh(原始增益预测))
粗恢复图 = 雾图 × 颜色增益 + 0.50 × 残差
去雾图   = clamp(粗恢复图 + 0.25 × 细化修正, 0, 1)
```

颜色增益是跨空间位置共享的三个数值；残差与细化修正则是与图像同尺寸的三通道特征图。增益由模型根据当前输入预测，不同图像可以获得不同的颜色调整。

**2. 对去雾图进行分割。** 去雾 RGB 图经过 ImageNet 归一化后进入 LiteAttentionUNet。深度可分离卷积用于轻量特征提取，attention gate 利用解码器上下文筛选编码器的跳跃连接特征。主配置启用 attention，关闭 SE 与辅助头，最终分类器将每个像素判定为道路或背景。

**3. 分阶段学习两个任务。** 首先预训练去雾网络，建立图像恢复映射；随后冻结去雾器，让分割网络适应去雾图；最后通过联合微调，让恢复损失与分割损失沿连接的网络共同优化。各阶段更新哪些参数，见 [60 / 20 / 20 训练安排](#training)。

<details>
<summary><strong>展开论文中的两个分支网络架构图</strong></summary>

**图 3 · ColorAwareUNet：** 全局 RGB 增益、残差预测与细化模块。

[![论文图 3：ColorAwareUNet 网络架构。](docs/assets/colorawareunet.png)](docs/assets/colorawareunet.png)

**图 4 · LiteAttentionUNet：** 轻量卷积、attention gate 与可选 SE 模块。

[![论文图 4：LiteAttentionUNet 网络架构。](docs/assets/liteattentionunet.png)](docs/assets/liteattentionunet.png)

以上图片直接取自提供的 Word 论文，点击可查看原始分辨率。可选模块的默认开关见下方配置表。

</details>

### 支持的模型

| 名称 | 内容 |
|---|---|
| `coloraware` | 论文主去雾模型 ColorAwareUNet |
| `c2pnet` | 仓库中的 C2PNet 实现 |
| `dcp` | 可训练的 DCP 恢复/细化＋LiteAttentionUNet，支持联合训练；保留无需权重的经典 DCP 推理 |
| `ffanet` | 仓库中的 FFA-Net 实现 |
| `grid` | 仓库中的 GridDehazeNet 实现 |
| `psd` | 仓库中的 PSDDehazeNet 实现 |

联合模型比较使用各方法**同一份 joint 权重中的去雾器和自带 LiteAttentionUNet**。如需单独开展共用分割器比较，通过 `--segmenter-checkpoint` 指定统一冻结权重，并将该协议的结果单独报告。神经网络对比模型沿用本仓库实现，不宣称与原作者官方代码、权重或结果完全一致。公开名称统一为 `dcp`：训练默认使用可学习细化扩展，不传权重推理时使用经典恢复，历史权重保持原实现。结果分别标记为 `learned-refinement`、`classical` 和 `historical-enhanced`。论文中应将可训练扩展作为可学习 DCP 变体报告。

### 论文主配置

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

### 训练目标

```text
去雾：L1 + 0.40 × SSIM_loss + 0.05 × VGG_perceptual
分割：CE + soft Dice
联合：去雾损失 + 分割损失
```

去雾损失为 `L1 + 0.40 × SSIM_loss + 0.05 × VGG_perceptual`；SSIM 使用 11×11 均值窗口，损失为 `(1 − SSIM) / 2`。感知损失使用冻结的 ImageNet VGG16 第 3、8、15 层特征。分割损失为 CE＋类别平均 soft Dice，联合阶段两项任务权重均为 1。

MSE、梯度差、ΔSat 和 CRerr 不参与默认优化；侧输出只用于兼容和可视化。局部增益与只放大增益属于消融设置。全局增益贡献热图表示 `mean(abs(input × (gain−1)))`，不是空间增益图。

### 论文组件与代码对应

| 组件 | 源码 |
| :--- | :--- |
| RGB 增益、残差与细化 | [`ColorAwareUnet.py`](dehaze_seg/models/ColorAwareUnet.py) |
| Attention 分割网络 | [`LiteAttentionUnet.py`](dehaze_seg/models/LiteAttentionUnet.py) |
| 联合网络与输出适配 | [`joint.py`](dehaze_seg/models/joint.py) · [`registry.py`](dehaze_seg/models/registry.py) |
| 训练目标 | [`losses.py`](dehaze_seg/losses.py) |
| 三阶段优化 | [`engine/train.py`](dehaze_seg/engine/train.py) |
| 消融配置 | [`ablations.py`](dehaze_seg/experiments/ablations.py) |

<a id="training"></a>

## 训练

`python train.py` 与 `python -m dehaze_seg train` 接受相同参数，无需在 `train.py` 后重复添加 `train`。使用 `python train.py --help` 查看完整选项。

### 三阶段训练

| 阶段 | 轮数 | 更新参数 | 优化目标 |
| :--- | :---: | :--- | :--- |
| 1 · 去雾预训练 | **60** | ColorAwareUNet | 去雾损失 |
| 2 · 分割训练 | **20** | LiteAttentionUNet；冻结去雾器 | 分割损失 |
| 3 · 联合微调 | **20** | 两个网络 | 两项损失之和 |

### 道路数据联合训练

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --output runs/paper --device cuda --amp
```

<details>
<summary><strong>展开论文主配置的完整命令</strong></summary>

```bash
python train.py --dataset paired-road --data-root datasets --model coloraware --pretrain-dehaze-epochs 60 --pretrain-seg-epochs 20 --finetune-epochs 20 --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/paper-explicit --device cuda --amp
```

</details>

<details>
<summary><strong>展开较小规模的 CPU 流程检查</strong></summary>

该命令将输入缩小至 64×64，每阶段运行 1 轮，并显式关闭感知损失以避免下载 VGG 权重。它会读取已准备的数据集并生成独立结果，用于检查环境与流程，不用于复现论文指标。

```bash
python train.py --data-root datasets --device cpu --resize 64 64 --batch-size 2 --pretrain-dehaze-epochs 1 --pretrain-seg-epochs 1 --finetune-epochs 1 --lam-perc 0 --output runs/smoke --no-plots
```

</details>

### 对比模型

在同一道路划分上训练神经网络去雾对比模型，再接入共用冻结分割器评估：

```bash
python train.py --dataset paired-road --data-root datasets --model c2pnet --pretrain-seg-epochs 0 --finetune-epochs 0 --output runs/comparison --device cuda --amp
```

该命令只更新去雾器；其权重中未训练的分割器需要在下方比较命令中通过 `--segmenter-checkpoint` 替换。省略两个零轮数选项可开展独立联合训练实验。

### DCP 去雾与分割联合训练

`train.py --model dcp` 现在同时支持训练**去雾细化分支**和 **LiteAttentionUNet**。道路数据仍需 `hazy/`、`clear/`、`masks/` 三元组。默认 `--dcp-mode learned` 使用仓库已有的 DCP 解析恢复、有界恢复、可学习细节细化头及细节/锐化处理。解析先验本身没有学习参数；细化头包含 22,563 个可训练参数，属于去雾分支。

```bash
python train.py --model dcp --dataset paired-road --data-root datasets --output runs/dcp-joint --device cuda --amp
```

| 阶段 | 默认轮数 | 更新分支 | 优化目标 |
| :--- | :---: | :--- | :--- |
| `dehaze` | 60 | DCP 可学习细化 | L1＋0.40×SSIM loss＋0.05×VGG 感知损失 |
| `seg` | 20 | LiteAttentionUNet；冻结去雾器 | CE＋Dice |
| `joint` | 20 | 两个可训练分支 | 去雾＋分割损失；分割梯度可传回去雾器 |

三个轮数参数恢复正常含义：`--pretrain-dehaze-epochs`、`--pretrain-seg-epochs`、`--finetune-epochs`。默认去雾损失需要 VGG 权重。SOTS 格式数据和 HSTS 在 `--dcp-mode learned` 下只训练去雾分支，无需分割标注；训练与测试目录仍需按下方说明隔离。

权重保存到 `runs/dcp-joint/paired-road/dcp/`，包含 `dehaze/`、`seg/`、`joint/` 三个阶段以及根目录的 `best.pth`、`last.pth`。各阶段按主训练器对应指标选择最优模型。配置中记录 `dehazer_type=learned-dcp`、细化参数及实际阶段安排，加载权重时会自动恢复可训练结构。

```bash
python -m dehaze_seg predict --checkpoint runs/dcp-joint/paired-road/dcp/best.pth --input datasets/hazy/001.png --output results/dcp-joint-predict
python -m dehaze_seg evaluate --checkpoint runs/dcp-joint/paired-road/dcp/best.pth --dataset paired-road --data-root datasets --split val --output results/dcp-joint-eval
```

之前的固定 DCP、只训练分割器模式仍可显式选择：

```bash
python train.py --model dcp --dcp-mode classical --dataset paired-road --data-root datasets --output runs/dcp-seg --device cuda --amp
```

此可选模式仅训练分割器，轮数为 `pretrain-seg-epochs + finetune-epochs`（默认 40）；不使用去雾预训练，也无需 VGG。该模式的旧权重仍按固定经典 DCP 加载。新的联合实验应使用新输出目录并重新训练。

联合训练属于额外的可学习 DCP 实验。图 7 使用各自联合权重中的分割器。另做共用分割器比较时，按下方命令指定**同一个** `--segmenter-checkpoint`，并与联合模型结果分开报告。

<details>
<summary><strong>神经网络去雾训练：SOTS 格式数据与增强 HSTS</strong></summary>

```bash
python train.py --dataset sots-indoor --data-root datasets/ITS/train --val-root datasets/ITS/val --model coloraware --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/sots --device cuda --amp
python train.py --dataset sots-outdoor --data-root datasets/OTS/train --val-root datasets/OTS/val --model coloraware --resize 512 512 --batch-size 2 --lr 1e-4 --output runs/sots --device cuda --amp
python train.py --dataset hsts --data-root HSTS --train-repeats 4 --output runs/hsts --device cuda --amp
```

请准备相互独立的 ITS/OTS 训练和验证目录，各自包含 `hazy/` 与 `clear/` 或 `gt/`。上述路径仅演示目录组织，不代表仓库提供了官方划分。SOTS 测试图像放在 `SOTS/indoor`、`SOTS/outdoor`，不参与训练或模型选择。`--dataset sots-*` 选择的是配对适配器，不表示可以在 SOTS 测试集上训练。省略 `--val-root` 时，按清晰参考图的 SHA-256 分组划分，同一场景的不同雾图进入同一组；此时验证比例作用于场景组。

基准数据集只执行 `--pretrain-dehaze-epochs`，使用去雾监督。示例采用 512×512 输入、1e-4 学习率、batch size 2 和去雾 60 轮。HSTS 命令属于内部增强实验，应评估保存的验证划分，不能对训练目录全量评估后作为独立测试成绩。`--train-repeats 4` 为可调整示例，不是官方 HSTS 协议。

</details>

### 常用参数

| 选项 | 用途 / 默认值 |
| :--- | :--- |
| `--pretrain-dehaze-epochs`、`--pretrain-seg-epochs`、`--finetune-epochs` | 三阶段轮数：60 / 20 / 20 |
| `--resize H W`、`--batch-size` | 输入尺寸与批大小：512 512 / 2 |
| `--seed`、`--val-ratio` | 划分种子与验证比例：42 / 0.15 |
| `--val-root` | 独立验证目录；此时训练根目录中的全部样本用于训练 |
| `--split-unit` | 道路/SOTS 默认 `scene`，HSTS 默认 `sample`；`sample` 使用历史按 ID 划分 |
| `--device`、`--amp` | `auto`、`cpu` 或 `cuda`；按需启用 CUDA 混合精度 |
| `--workers`、`--threads` | 数据加载进程：0；CPU 线程：4 |
| `--use-se`、`--no-attention`、`--no-imagenet-norm` | 修改分割网络的默认配置 |
| `--output` | 新实验的输出父目录 |

`--dcp-mode learned` 是 DCP 训练默认值，在道路数据上使用正常的 60 / 20 / 20 安排；`--dcp-mode classical` 则选择可选的固定去雾器、仅训练分割器流程。

### 训练输出

```text
runs/paper/paired-road/coloraware/
├── config.json          # 完整模型、训练和实验配置
├── split.json           # 样本 ID、根目录、分组规则与清晰参考图哈希
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

<a id="prediction-and-evaluation"></a>

## 推理与评估

### 单图与批量预测

```bash
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy/001.png --output results/single
python -m dehaze_seg predict --checkpoint runs/paper/paired-road/coloraware/best.pth --input datasets/hazy --output results/batch
```

输出包括去雾图、类别编号 PNG、分割叠加图和拼图；纯去雾权重指定 `--segmenter-checkpoint` 后也可输出分割结果。神经网络输入内部补齐至 16 的倍数，经典 DCP 使用未补齐的图像。预测保留原图尺寸；可用 `--resize H W` 降低推理开销，结果会插值回原尺寸。

### 数据集评估

评估围绕论文中的三个角度展开。不同指标反映输出的不同性质，应结合图像恢复质量与分割区域重合程度一起分析。

| 评估角度 | 指标 | 含义 |
| :--- | :--- | :--- |
| 图像恢复质量 | PSNR、SSIM ↑ | 相对于清晰参考图的像素保真度与结构相似性 |
| 颜色保持 | ΔSat、CRerr ↓ | 饱和度与 RGB 通道相对比例的偏差 |
| 道路分割 | mIoU、平均 Dice、F1 ↑ | 预测区域与真实标注的重合程度 |

↑ 表示越大越好，↓ 表示越小越好。分割指标需要真实标注，在配对道路数据上计算。

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/road-eval
python -m dehaze_seg evaluate --checkpoint runs/sots/sots-indoor/coloraware/best.pth --dataset sots-indoor --data-root SOTS/indoor --split all --output results/sots-eval
```

对已知训练或验证根目录，评估默认使用保存的验证划分；独立基准测试目录默认 `all`，道路评估默认 `val`。可通过 `--split-file` 明确指定划分。指定共用分割器时，默认从它的权重中读取划分，所有对比模型使用同一有序样本列表。权重若使用独立 `--val-root` 训练，验证时会提示使用该目录。

评估会拒绝与去雾器或共用分割器训练数据存在已知重叠的样本，包括清晰参考文件哈希相同的复制数据。`--allow-training-overlap` 仅用于显式诊断，并在结果中记录重叠数量。这些检查不能证明未知训练历史的独立性，也无法识别所有经过变换或重新编码的副本。旧权重缺少划分清单时，会按 seed/val_ratio 重建请求的划分，要求数据内容未变；重建本身不构成独立测试集。

`--metric-align crop` 默认将尺寸不一致的预测与参考图中心裁剪到共同区域；`resize` 将参考图缩放到预测尺寸；`none` 要求尺寸完全相同。评估默认 `--metric-resolution original`：即使设置了 `--resize`，也会先放回原图尺寸再评分。要核对道路训练验证日志，须使用同一权重、同一划分，并**同时指定** `--resize 512 512 --metric-resolution inference`。此时直接在模型推理网格上评分，参考 RGB 使用 PIL 双线性缩放，标注使用最近邻缩放，与道路训练加载器一致：

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --resize 512 512 --metric-resolution inference --output results/road-eval-512
```

**指标定义：** `miou` 和 `mdice` 从同一个 `argmax` 硬标签预测及混淆矩阵计算，对背景（类别 0）和道路（类别 1）取平均；`iou_class_1`、`dice_class_1` 仅表示前景。硬标签 `mdice` 不等于 `1 − soft Dice loss`。在预测和标注中均不存在的类别计为 0。同一掩码、类别和汇总方式下，必有 `mdice >= miou`；将前景 Dice 与宏平均 mIoU 混用则不满足此前提。论文单图标注应读取同一逐图记录里的两列，不应混入全数据汇总值或训练损失。

评估保存逐图 `metrics.csv`（含逐类别 IoU/Dice）、汇总 `summary.json`，并在 `protocol.json` 记录样本 ID、模型配置、权重哈希、清晰参考、指标定义、评分尺寸和共用分割器来源。`segmentation_metrics.json` 保留逐图混淆矩阵、标注哈希及实际评分尺寸，便于独立复算。恢复指标按图平均；数据集分割指标先累加混淆矩阵再计算，因此不一定等于 CSV 逐图指标的平均值。验证 PSNR 也按图平均。`--save-images` 始终保存原图尺寸的展示图片，即使指标在推理尺寸上计算；`--limit 1` 可只检查一张。

### 多模型比较

以下为可选的共用冻结分割器比较，与图 7 的完整联合模型协议分别报告：

```bash
python -m dehaze_seg evaluate --checkpoint runs/paper/paired-road/coloraware/best.pth runs/comparison/paired-road/c2pnet/best.pth --include-dcp --segmenter-checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --save-images --output results/comparison
```

共用分割器只加载一次，冻结参数并保持 eval 模式；所有去雾方法使用它保存的归一化配置，包括纯去雾权重和经典 DCP。结果记录该权重路径及 SHA-256；各模型自带的分割器会被替换。省略该参数则使用各权重自己的分割器，属于另一种协议。请在比较前固定共用权重，不要依据测试结果选择它。历史共用权重必要时可通过 `--segmenter-model`、`--segmenter-legacy-profile`、`--segmenter-model-config` 补充配置。

经典 DCP 也可以单独运行，公开名称保持为 `dcp`：

```bash
python -m dehaze_seg predict --model dcp --input datasets/hazy/001.png --output results/dcp
python -m dehaze_seg evaluate --model dcp --segmenter-checkpoint runs/paper/paired-road/coloraware/best.pth --dataset paired-road --data-root datasets --split val --output results/dcp-eval
```

<a id="checkpoints"></a>

## 权重加载

新权重保存完整 `model_config`、训练配置及实际数据划分，包含根目录与清晰参考图哈希。加载时自动恢复结构和数值配置。支持历史 `model_state`、`model_state_dict`、`state_dict`、`dehazer_state` 字典，以及常见外层前缀。历史 `args` 会映射为对应配置。旧 `gain_local` 权重保留原来的增益下限，订正后的默认值只用于新实验。

```bash
python -m dehaze_seg predict --checkpoint weights/colorawareunet.pth --input datasets/hazy/001.png --resize 512 512 --output results/legacy
```

上述权重路径仅表示已有的本地文件，仓库不附带权重下载。无配置的裸参数文件需要明确选择历史设置：

```bash
python -m dehaze_seg predict --checkpoint weights/raw.pth --model coloraware --legacy-profile paper --input datasets/hazy/001.png --output results/raw
```

`paper` 使用论文默认配置，`legacy-infer` 对应旧通用推理入口的默认配置（例如 ColorAwareUNet gain=0.65、无 gain 下限）。对于其他历史结构，用 `--model-config model.json` 提供完整模型配置；格式与训练 `config.json` 中的 `model_config` 对象一致。结构不匹配会明确报错，不会部分加载后继续推理。旧模块路径和旧命令入口不再保留。

<a id="ablations-and-visualization"></a>

## 消融与可视化

### 注册式消融

```bash
python -m dehaze_seg ablate --list
```

<details>
<summary><strong>展开结构、增益与注意力消融命令</strong></summary>

```bash
python -m dehaze_seg ablate --data-root datasets --experiments arch_baseline_unet,arch_no_color_gain,arch_no_refine,arch_full --output runs/ablation --device cuda --amp
python -m dehaze_seg ablate --data-root datasets --experiments gain_local,gain_amp,gain_no_min --output runs/gain-ablation --device cuda --amp
python -m dehaze_seg ablate --data-root datasets --experiments attention_none,attention_gate,attention_se,attention_both --output runs/attention-ablation --device cuda --amp
```

</details>

还提供训练策略、损失、模型容量、数据增强、输入归一化和直接雾图分割实验。`--experiments all` 运行完整注册表。所有消融共享训练流程、随机划分和指标；结果按实验名称保存，并汇总到 `ablation_summary.csv`。

### 复现实验说明

| 设置 | 当前实现与适用范围 |
| :--- | :--- |
| 表 4 局部增益 | `gain_local`：局部 tanh、scale=0.30，**关闭下限**（`gain_min=None`） |
| 表 4 只放大增益 | 保留 `1 + 0.30 × tanh(ReLU(raw))`；增益末层权重初始化为零，偏置为 **0.001**，使梯度能通过 ReLU。主模型 tanh 仍使用零初始化。 |
| 历史只放大实验 | 末层全零初始化会让增益头无法学习。加载仍保留已保存的权重；订正实验需要重新训练。论文中统一零初始化的表述需要补充这一例外。 |
| 弱 / 强增益 | 仓库设置为 `(scale, min)=(0.10, 0.98)` / `(0.50, 0.90)`；论文未明确这些具体数值。局部平滑核为 15。 |
| 表 6 attention / SE | 已注册四种组合。表中 5.878 / 5.944 / 6.097 / 6.164 M 对应**联合模型**参数量，不是单独分割器；当前每个变体执行完整三阶段训练。 |
| 消融汇总 | `ablation_summary.csv` 记录末轮验证指标，不是重新评估 `best.pth` 的结果；每阶段从前阶段末轮参数继续训练。 |

论文尚未完整说明注意力实验的训练预算、冻结范围和全部模型选择设置；声称数值复现前，需要对照原始日志确认。训练器各阶段记录活动损失，在仅训练分割器时不会额外记录去雾训练损失，因此仅凭当前 CSV 不能完整重绘图 6(a)。CPU 回归检查已覆盖订正流程，六个道路模型另通过 CUDA 推理核验；未重新执行完整训练及全部论文实验。

### 可视化

```bash
python -m dehaze_seg visualize gain --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --samples 001 --output-dir results/color_gain
python -m dehaze_seg visualize components --checkpoint runs/paper/paired-road/coloraware/best.pth --data-root datasets --sample 001 --output-dir results/components
python -m dehaze_seg visualize curves --metrics runs/paper/paired-road/coloraware/metrics.csv --output results/curves
python -m dehaze_seg visualize introduction --data-root datasets --sample 001 --visualization-dir results/color_gain --output-dir results/figures
```

增益和组件可视化要求联合 ColorAwareUNet 权重。Introduction 拼图使用前一步生成的相同 sample 结果。各子命令可通过 `--help` 查看选项。

### 重新生成分割对比图（图 7）

```bash
python generate_figure7.py --segmentation-protocol joint --checkpoint ckpt_datasets_joint/dcp/joint/best.pth ckpt_datasets_joint/ffanet/joint/best.pth ckpt_datasets_joint/grid/joint/best.pth ckpt_datasets_joint/psd/joint/best.pth ckpt_datasets_joint/coloraware/joint/best.pth ckpt_datasets_joint/c2pnet/joint/best.pth --data-root datasets --sample 003 --resize 512 512 --output results/figure7 --device cuda
```

该命令生成**八栏**：雾图、DCP、FFA-Net、GridDehazeNet、PSD、ColorAwareUNet、C2PNet 和 GT。每种方法使用其 **joint/best 完整联合权重及自带分割器**。脚本要求六种方法齐全，缺少权重时在推理前报错，不生成不完整的图 7。以上路径对应历史实验目录，请按实际权重位置修改；`python -m dehaze_seg visualize segmentation` 提供相同入口。没有可用 CUDA 时使用 `--device cpu`。

`003` 是所给历史权重验证划分中的首个 ID。换用其他权重时，省略 `--sample` 可采用该权重的首个验证 ID，或指定其保存划分中的样本。程序拒绝训练样本及已知参考图重叠。该图是单个验证样本示例，不是独立测试集成绩，也不表示复现了表 1。

图上标注为 **mIoU / mDice**，从同一张完整 512×512 硬标签掩码计算，包含背景与前景，并直接读取本次指标记录生成。红框放大只用于展示，不改变评分区域；展示图片保持原图宽高比。`--roi X0 Y0 X1 Y1` 为所有栏设置相同的归一化裁剪范围。输出包括 300 dpi 的 `figure7.png`、`figure7.pdf`、原始预测/GT 掩码、混淆矩阵、权重与输入哈希、CSV/JSON 指标及图注。

图中标题仅保留模型名称。指标数字及 `mIoU / mDice` 使用 **Times New Roman（新罗马）**。Linux 等环境若未安装该字体，可传入 `--metric-font /path/to/times.ttf`；字体缺失时会明确报错。

历史 DCP 权重对应仓库中的增强实现，具体版本保留在图注及 JSON/CSV 记录中。如需另做共用分割器实验，显式传入 `--segmentation-protocol shared-frozen --segmenter-checkpoint PATH`；只有该协议支持移除 DCP 权重并使用 `--include-dcp` 加入**无需权重的经典 DCP**，其余五个权重仍须提供。joint 模式会拒绝共用分割器选项，避免意外替换配套分割器。不会用旧图标注或分数补齐缺失权重。生成图片及本地权重保持 Git 忽略，生成代码纳入公开仓库。

<a id="project-structure"></a>

## 代码结构

```text
train.py             # 直接训练入口
generate_figure7.py   # 六模型完整分割对比图
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

统一入口 `python -m dehaze_seg` 或安装后的 `dehaze-seg` 提供 `train`、`predict`、`evaluate`、`ablate`、`visualize` 五个子命令。可选标注和数据整理工具见[工具说明](tools/README.zh-CN.md)。Git 忽略本地数据、权重、生成结果、论文文件和 notebook，正常跟踪源码与测试。

<a id="validation"></a>

## 验证范围

```bash
python -m unittest discover -s tests -v
```

在[安装说明](#installation)列出的本机环境中，**29 项 CPU 回归测试通过**，覆盖模型前向、只放大增益的实际学习、attention/SE 变体、数据配对、同步增强、阶段冻结、参数更新、严格权重往返加载、场景隔离、共用冻结分割器、经典 DCP 分割训练，以及可学习 DCP 去雾/联合训练。联合 DCP 测试还验证了分割损失能够传回去雾细化头。图 7 测试覆盖六种方法齐全、缺失权重报错、模型配置可序列化，以及保存掩码、混淆矩阵和图中分数一致。测试不会下载数据或 VGG 权重。

另已检查 185 组道路三元组、小样本三阶段训练和推理/评估/消融/可视化流程。六个历史道路模型均通过严格加载，并在原 28 张验证图上分别核验自身分割器与同一个冻结分割器的结果；重算的自身 mIoU/mDice 与权重记录相差均小于 0.000018。其中两张验证图与训练集共用清晰参考，另行报告了排除它们后的 26 张子集。上述检查验证了所给权重，不代表建立了独立测试集或复现全部论文表格。后续已在 RTX 3080 Ti（PyTorch 2.3.0+cu121）上核验 18 份阶段权重的历史验证集与两种推理尺寸；图 7 默认采用各自完整 joint best 模型。尚未重新执行完整训练。

<a id="faq"></a>

## 常见问题

<details>
<summary><strong>VGG 下载失败</strong></summary>

默认感知损失需要 `vgg16-397923af.pth`，放在 `TORCH_HOME/hub/checkpoints/` 缓存中；未设置 `TORCH_HOME` 时通常为 `~/.cache/torch/hub/checkpoints/`。也可显式 `--lam-perc 0` 做无感知损失实验，不能将其当作论文完整目标。

</details>

<details>
<summary><strong>CPU 很慢或显存不足</strong></summary>

缩小 `--batch-size`、`--resize`；CPU 可调整 `--threads`。缩小训练尺寸会改变实验设置。

</details>

<details>
<summary><strong>找不到数据</strong></summary>

检查解压后是否多嵌套了一层 `datasets/`，并核对文件主名和目录名。

</details>

<details>
<summary><strong>模型不匹配</strong></summary>

提供正确旧配置或对应权重，不要通过非严格加载掩盖结构差异。

</details>

<a id="citation"></a>

## 论文与引用

本仓库对应页首论文的方法实现。实验结论与评估协议请以最终论文为准；回归测试不能替代完整复现实验。以下条目用于引用软件仓库，论文正式发表后请同时使用出版社提供的论文引用信息。

```bibtex
@misc{colorawarenet_joint_model,
  title = {ColorAwareNet Joint Model: Color-Gain-Guided Image Dehazing and Semantic Segmentation},
  howpublished = {GitHub repository},
  url = {https://github.com/dongfangshiwen/colorawarenet-joint-model}
}
```

问题反馈请提交至 [GitHub Issues](https://github.com/dongfangshiwen/colorawarenet-joint-model/issues)，并附上运行命令、环境版本与错误日志。

---

<div align="center">

[返回顶部](#colorawarenet) · [English](README.md)

</div>
