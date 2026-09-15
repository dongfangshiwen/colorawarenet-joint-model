# Optional dataset tools / 可选数据工具

Run these scripts from the repository root. Install `pip install -e ".[tools]"` first.
从仓库根目录运行；图像工具需要 OpenCV，交互标注工具还需要 PySide6 和支持桌面窗口的环境。

- `python tools/interactive_mask_maker.py --help`: interactive image annotation / 交互标注。
- `python tools/synth_realistic_fog.py --help`: synthesize haze / 合成雾图。
- `python tools/remove.py --root datasets --dirs hazy,masks --suffix _fog_v1 --dry-run`: preview suffix removal / 预览移除文件名后缀。
- `python tools/rename.py --help`: preview or apply sequential renaming / 预览或执行顺序重命名。

Image, clear-reference and mask stems must stay aligned. These utilities are optional;
training never invokes them. 标注、清晰图与雾图需保持同名对应，训练过程不会自动运行这些工具。
