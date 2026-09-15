# Dataset tools

**English** · [简体中文](README.zh-CN.md) · [Project overview](../README.md)

Optional utilities for annotation, haze synthesis and filename organization. Run them from the repository root after installing the extra dependencies:

```bash
python -m pip install -e ".[tools]"
```

Image tools use OpenCV. Interactive annotation also needs PySide6 and an environment that can display desktop windows.

| Tool | Purpose | Usage |
| :--- | :--- | :--- |
| [`interactive_mask_maker.py`](interactive_mask_maker.py) | Interactive image annotation | `python tools/interactive_mask_maker.py --help` |
| [`synth_realistic_fog.py`](synth_realistic_fog.py) | Synthesize hazy images | `python tools/synth_realistic_fog.py --help` |
| [`rename.py`](rename.py) | Preview or apply sequential renaming | `python tools/rename.py --help` |
| [`remove.py`](remove.py) | Remove a filename suffix | Preview command below |

Preview suffix removal before applying changes:

```bash
python tools/remove.py --root datasets --dirs hazy,masks --suffix _fog_v1 --dry-run
```

Hazy images, clear references and segmentation masks must keep matching stems after file operations. Training does not invoke these tools automatically.
