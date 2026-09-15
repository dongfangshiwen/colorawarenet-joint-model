# 数据集工具

[English](README.md) · **简体中文** · [返回项目说明](../README.zh-CN.md)

用于标注、合成雾图与整理文件名的可选工具。先安装额外依赖，然后从仓库根目录运行：

```bash
python -m pip install -e ".[tools]"
```

图像工具使用 OpenCV；交互标注还需要 PySide6 和能够显示桌面窗口的环境。

| 工具 | 用途 | 使用方式 |
| :--- | :--- | :--- |
| [`interactive_mask_maker.py`](interactive_mask_maker.py) | 交互式图像标注 | `python tools/interactive_mask_maker.py --help` |
| [`synth_realistic_fog.py`](synth_realistic_fog.py) | 合成雾图 | `python tools/synth_realistic_fog.py --help` |
| [`rename.py`](rename.py) | 预览或执行顺序重命名 | `python tools/rename.py --help` |
| [`remove.py`](remove.py) | 移除文件名后缀 | 见下方预览命令 |

移除文件名后缀前，先预览变更：

```bash
python tools/remove.py --root datasets --dirs hazy,masks --suffix _fog_v1 --dry-run
```

文件整理后，雾图、清晰图与分割标注必须保持文件主名对应。训练过程不会自动运行这些工具。
