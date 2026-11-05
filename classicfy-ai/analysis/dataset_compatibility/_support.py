"""Portable paths and plotting fonts for the dataset compatibility reports."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS = Path(os.environ.get("CLASSICFY_DATASETS", REPO_ROOT.parent / "datasets")).expanduser().resolve()


def source_label(path):
    path = Path(path).resolve()
    for root, prefix in [(DATASETS, "datasets"), (REPO_ROOT, "repository")]:
        try:
            return str(Path(prefix) / path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def setup_fonts():
    from matplotlib import font_manager, rcParams
    mac_font = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if mac_font.is_file():
        font_manager.fontManager.addfont(str(mac_font))
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", "NanumGothic"]:
        if name in available:
            rcParams["font.family"] = name
            return
    raise RuntimeError("Korean plotting font required: install Noto Sans CJK KR, NanumGothic, or Malgun Gothic.")


def clean_svg_outputs(directory):
    """Keep generated SVGs free of insignificant trailing whitespace."""
    for path in Path(directory).glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
