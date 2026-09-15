"""
pipeline/elements.py — RawElement -> KnowledgeUnit.

Per-type conversion:
    text/paragraph  -> TEXT unit (sub-split when longer than CHUNK_SIZE)
    title/heading   -> small TEXT unit carrying the section title
    list            -> TEXT unit preserving bullet structure
    code            -> CODE unit
    equation        -> EQUATION unit
    table           -> TABLE unit (structured {headers, rows} + embedded prose)
    image           -> saved to images/ then IMAGE unit (type refined via vision)
"""

from __future__ import annotations

import re

from config import CHUNK_SIZE, CHUNK_OVERLAP, IMAGES_DIR
from knowledge_unit import KnowledgeUnit, KnowledgeUnitType
from pipeline.parser import RawElement

TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]+\|?$")


def _split_long_text(text: str) -> list[str]:
    """Sub-split a long section into ~CHUNK_SIZE chunks with overlap."""
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            # try to break at a newline
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl
        chunks.append(text[start:end].strip())
        start = max(end - CHUNK_OVERLAP, start + 1)
        if start >= len(text):
            break
    return [c for c in chunks if c]


def parse_markdown_table(rows: list[str]) -> dict:
    """Convert markdown table rows into {headers, rows}."""
    clean: list[list[str]] = []
    for line in rows:
        line = line.strip()
        if not line.startswith("|"):
            line = "|" + line
        if not line.endswith("|"):
            line = line + "|"
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not any(cells):  # separator row like |---|---|
            continue
        if TABLE_SEPARATOR_RE.match(line) and all(c in {"", ":", "-", " "} for c in cells):
            continue
        clean.append(cells)
    if not clean:
        return {"headers": [], "rows": []}
    headers = clean[0]
    rows_data = clean[1:]
    return {"headers": headers, "rows": rows_data}


def table_to_prose(table: dict) -> str:
    """Retrieval-friendly textual representation of a structured table."""
    headers, rows = table["headers"], table["rows"]
    out: list[str] = []
    if headers:
        out.append("Table columns: " + ", ".join(headers) + ".")
    for r in rows:
        if len(headers) == len(r) and headers:
            pairs = [f"{h} is {v}" for h, v in zip(headers, r) if str(v).strip()]
            if pairs:
                out.append(" ".join(pairs) + ".")
        else:
            out.append("Row: " + " | ".join(str(c) for c in r) + ".")
    return "\n".join(out)


def build_units_from_elements(elements: list[RawElement], source_doc: str) -> list[KnowledgeUnit]:
    """Convert raw parsed elements into final KnowledgeUnits (images not yet described)."""
    units: list[KnowledgeUnit] = []

    for el in elements:
        path = list(el.section_path)

        if el.kind in {"title", "text"}:
            sub = _split_long_text(el.content)
            for part in sub:
                units.append(KnowledgeUnit(
                    type=KnowledgeUnitType.TEXT,
                    content=part,
                    embedding_text=part,
                    page=el.page,
                    source_document=source_doc,
                    section_path=path,
                ))

        elif el.kind == "list":
            units.append(KnowledgeUnit(
                type=KnowledgeUnitType.LIST,
                content=el.content,
                embedding_text=el.content,
                page=el.page,
                source_document=source_doc,
                section_path=path,
            ))

        elif el.kind == "code":
            units.append(KnowledgeUnit(
                type=KnowledgeUnitType.CODE,
                content=el.content,
                embedding_text=f"Code block:\n{el.content}",
                page=el.page,
                source_document=source_doc,
                section_path=path,
            ))

        elif el.kind == "equation":
            units.append(KnowledgeUnit(
                type=KnowledgeUnitType.EQUATION,
                content=el.content,
                embedding_text=f"Mathematical equation: {el.content}",
                page=el.page,
                source_document=source_doc,
                section_path=path,
            ))

        elif el.kind == "table":
            structured = parse_markdown_table(el.content.split("\n"))
            prose = table_to_prose(structured)
            title = path[-1] if path else ""
            embed = f"{title}.\n{prose}" if title else prose
            units.append(KnowledgeUnit(
                type=KnowledgeUnitType.TABLE,
                content=el.content,            # original markdown table
                embedding_text=embed,           # prose form
                page=el.page,
                source_document=source_doc,
                section_path=path,
                structured_data=structured,
            ))

        elif el.kind == "image":
            units.append(KnowledgeUnit(
                type=KnowledgeUnitType.IMAGE,
                content=f"image ref: {el.content}",   # placeholder; resolved below
                embedding_text=el.content,            # placeholder; replaced by VLM description
                page=el.page,
                source_document=source_doc,
                section_path=path,
                image_path=None,
            ))

    return units


def resolve_images(
    units: list[KnowledgeUnit],
    extract,
) -> list[KnowledgeUnit]:
    """Write extracted images to disk and attach their paths to image units."""
    resolved: list[KnowledgeUnit] = []
    for u in units:
        if u.type is not KnowledgeUnitType.IMAGE:
            resolved.append(u)
            continue
        ref = u.embedding_text
        img = extract.images.get(ref)
        if img is None:
            # try to locate the largest image as a fallback (rare in practice)
            candidates = [k for k in extract.images if u.page in _page_of(k, extract)]
            img = extract.images.get(candidates[0]) if candidates else None
        if img is not None:
            saved = _save(extract, img, ref, u.page)
        else:
            saved = None
        u.image_path = saved
        u.content = saved or f"image ref: {ref}"
        resolved.append(u)
    return resolved


def _page_of(ref_name: str, extract) -> int:
    for pm in extract.pages:
        if ref_name in pm.image_refs:
            return pm.page
    return 0


def _save(extract, img, ref: str, page: int) -> str | None:
    from pipeline.extract_markdown import save_image

    return save_image(img, ref, IMAGES_DIR, page)