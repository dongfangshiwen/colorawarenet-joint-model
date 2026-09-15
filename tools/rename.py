"""Preview sequential renaming without changing image formats."""
import argparse
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--apply", action="store_true", help="Apply the displayed mapping; default is preview")
    args = parser.parse_args()
    folder = args.folder.resolve(strict=True)
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"})
    pairs = [(p, folder / f"{i:03d}{p.suffix.lower()}") for i, p in enumerate(images, 1)]
    sources = {p for p, _ in pairs}
    for source, target in pairs:
        if target.exists() and target not in sources:
            raise ValueError(f"Destination already exists: {target}")
        print(f"{source.name} -> {target.name}")
    if args.apply:
        pending = []
        for source, target in pairs:
            temporary = folder / (".rename-" + uuid.uuid4().hex + source.suffix)
            source.rename(temporary)
            pending.append((temporary, target))
        for temporary, target in pending:
            temporary.rename(target)


if __name__ == "__main__":
    main()
