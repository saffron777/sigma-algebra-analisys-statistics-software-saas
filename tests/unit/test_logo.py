"""The generated logo assets: present, well formed and at the sizes the application relies on."""

import io
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

IMG = Path(__file__).resolve().parent.parent.parent / "sigma" / "presentation" / "web" / "static" / "img"
SVGS = ["logo-mark.svg", "logo-mark-small.svg", "favicon.svg", "logo-wordmark.svg"]


@pytest.mark.parametrize("name", SVGS)
def test_svgs_are_valid_xml_without_scripts_or_external_references(name):
    text = (IMG / name).read_text(encoding="utf-8")
    root = ET.fromstring(text)
    assert root.tag.endswith("svg") and root.get("viewBox")
    lowered = text.lower()
    assert (
        "<script" not in lowered
        and "onload" not in lowered
        and "http://" not in lowered.replace("http://www.w3.org", "")
    )
    assert "href=" not in lowered  # nothing is fetched from anywhere


def test_the_full_mark_has_the_sigma_and_the_compact_one_does_not():
    assert "<polyline" in (IMG / "logo-mark.svg").read_text(encoding="utf-8")  # the Σ
    for name in ("logo-mark-small.svg", "favicon.svg"):
        assert "<polyline" not in (IMG / name).read_text(encoding="utf-8")
    assert (IMG / "favicon.svg").read_text(encoding="utf-8") == (IMG / "logo-mark-small.svg").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("name", "size"),
    [("logo-mark-512.png", (512, 512)), ("logo-mark-256.png", (256, 256)), ("apple-touch-icon.png", (180, 180))],
)
def test_png_marks(name, size):
    with Image.open(IMG / name) as image:
        assert image.format == "PNG" and image.size == size
        if name != "apple-touch-icon.png":
            assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0  # transparent corners
        else:
            assert image.getpixel((0, 0))[:3] == (255, 255, 255)  # opaque, as iOS wants


def test_wordmark_is_wider_than_tall():
    with Image.open(IMG / "logo-wordmark.png") as image:
        assert image.height == 256 and image.width > 2 * image.height
    assert (IMG.parent.parent.parent.parent.parent / "docs" / "logo.png").is_file()


def test_favicon_ico_carries_the_three_sizes():
    with Image.open(IMG / "favicon.ico") as icon:
        assert icon.format == "ICO" and set(icon.ico.sizes()) == {(16, 16), (32, 32), (48, 48)}


def test_regenerating_is_deterministic():
    import importlib.util

    spec = importlib.util.spec_from_file_location("make_logo", IMG.parents[4] / "scripts" / "make_logo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.mark_svg(module.FULL) == (IMG / "logo-mark.svg").read_text(encoding="utf-8")
    # PNG byte streams differ between Pillow/zlib builds; the pixels and dimensions are the contract.
    generated = Image.open(io.BytesIO(module.png_bytes(module.render_mark(module.FULL, 256))))
    with Image.open(IMG / "logo-mark-256.png") as saved:
        assert generated.mode == saved.mode and generated.size == saved.size
        assert generated.getpixel((0, 0))[3] == 0  # transparent corner
    assert io.BytesIO(module.png_bytes(module.render_mark(module.COMPACT, 64))).getbuffer().nbytes > 500
