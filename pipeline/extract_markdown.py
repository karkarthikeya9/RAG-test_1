"""
pipeline/extract_markdown.py — PDF -> Markdown.

Wraps the Markdown engine so downstream code never touches marker directly.
Swapping engines (e.g. marker -> docling) only changes this module.

Marker is used because it preserves document structure (headings, tables,
images, reading order) with high fidelity. It runs its ML models on CPU.

Output:
    ExtractResult.pages   -> [PageMarkdown(page=1, markdown="..."), ...]
    ExtractResult.images  -> {"ref_name": PIL.Image, ...}
"""

from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

os.environ["TORCH_DEVICE"] = "cpu"

from PIL import Image  # noqa: E402

from config import MARKDOWN_ENGINE, MARKDOWN_DIR  # noqa: E402

# page marker emitted by marker with paginate_output=True:
#   {0}------------------------------------------------
PAGE_MARKER_RE = re.compile(r"\n\n\{(\d+)\}\s*-{20,}\s*\n\n")

# markdown image reference   ![](some_name.jpeg)  or  ![alt](some_name.jpg)
IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\(\s*([^)\s]+)\s*\)")


@dataclass
class PageMarkdown:
    page: int              # 1-based
    markdown: str
    image_refs: list[str] = field(default_factory=list)


@dataclass
class ExtractResult:
    source_document: str   # original filename
    pages: list[PageMarkdown]
    images: dict           # ref name -> PIL.Image


def _run_marker(pdf_path: str) -> tuple[str, dict]:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered

    model_dict = create_model_dict()
    try:
        converter = PdfConverter(
            artifact_dict=model_dict,
            config={"paginate_output": True},
        )
        rendered = converter(pdf_path)
        markdown, ext, images = text_from_rendered(rendered)
        if ext != "md":
            raise ValueError(f"expected markdown, got {ext}")
        return markdown, images
    finally:
        try:
            from marker.models import shutdown_models

            shutdown_models(model_dict)
        except Exception:  # pragma: no cover
            pass


def _split_pages(markdown: str) -> list[tuple[int, str]]:
    """Split markdown on marker page markers -> [(page_0based, md), ...]."""
    parts = PAGE_MARKER_RE.split(markdown)
    # parts[0] is preamble before any marker; then pairs of (page_id, content)
    result: list[tuple[int, str]] = []
    # iterate: parts[1::2] are page ids, parts[2::2] are contents
    for i in range(1, len(parts) - 1, 2):
        page_id = int(parts[i])
        content = parts[i + 1].strip()
        result.append((page_id, content))
    if not result:
        result.append((0, markdown.strip()))
    return result


def _find_refs(page_md: str) -> list[str]:
    return IMAGE_REF_RE.findall(page_md)


def convert_pdf(pdf_path: str, out_dir: str | None = None) -> ExtractResult:
    """Convert one PDF to per-page markdown. Loads marker models."""
    pdf_path = str(Path(pdf_path))
    source = Path(pdf_path).name

    if MARKDOWN_ENGINE == "marker":
        markdown, images = _run_marker(pdf_path)
    else:
        raise ValueError(f"unknown MARKDOWN_ENGINE: {MARKDOWN_ENGINE!r}")

    raw_pages = _split_pages(markdown)
    pages: list[PageMarkdown] = []
    for page_id, md in raw_pages:
        refs = _find_refs(md)
        pages.append(PageMarkdown(page=page_id + 1, markdown=md, image_refs=refs))

    # persist intermediate markdown for inspectability
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w\-.]", "_", source) + ".md"
        (Path(out_dir) / safe).write_text(markdown, encoding="utf-8")

    return ExtractResult(source_document=source, pages=pages, images=images)


def save_image(img: Image.Image, ref_name: str, images_dir: str, page: int) -> str | None:
    """Save an extracted image to disk. Returns relative path or None."""
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    ext = Path(ref_name).suffix.lower() or ".png"
    if ext not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"
    # deterministic name derived from original ref-name
    base = Path(ref_name).stem
    safe = re.sub(r"[^\w\-.]", "_", base)
    target = Path(images_dir) / f"page_{page:02d}__{safe}{ext}"
    try:
        rgb = img.convert("RGB")
        rgb.save(target)
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] could not save image {ref_name!r}: {exc}")
        return None
    return str(target)