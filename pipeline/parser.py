"""
pipeline/parser.py — Markdown -> typed raw elements.

Turns marker-generated per-page Markdown into a stream of RawElements
(text / title / list / code / equation / table / image) each tagged with its
section_path (the active heading stack at that point in the document).

This is the "Markdown-aware structural chunking" step. Chunks break at
structural boundaries (headings, images, tables) instead of slicing through
markup like a character splitter would.

None of this code knows about marker — it only sees Markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pipeline.extract_markdown import PageMarkdown, IMAGE_REF_RE

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|")
FENCE_RE = re.compile(r"^```")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
EQUATION_RE = re.compile(r"^\$\$\s*$")


@dataclass
class RawElement:
    kind: str                        # title|text|list|code|equation|table|image
    page: int
    content: str
    section_path: list[str] = field(default_factory=list)

    def __repr__(self):  # pragma: no cover
        return f"<RawElement {self.kind} p{self.page} [{self.section_path}]>"


def _strip_literal_heading(text: str) -> str:
    """Marker may emit '## # Client Server Architecture'; collapse deduped #s."""
    t = text.strip()
    while t.startswith("#"):
        t = t[1:].strip()
    return t


def parse_page(page: PageMarkdown) -> list[RawElement]:
    lines = page.markdown.split("\n")
    elements: list[RawElement] = []
    heading_stack: list[str] = []

    def section_path():
        return list(heading_stack)

    def flush_text(acc: list[str], kind: str):
        text = "\n".join(acc).strip()
        if text:
            elements.append(RawElement(kind, page.page, text, section_path()))
        acc.clear()

    text_acc: list[str] = []
    table_acc: list[str] = []
    code_acc: list[str] = []
    list_acc: list[str] = []
    in_code = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # fenced code blocks
        if FENCE_RE.match(line):
            if not in_code:
                in_code = True
                code_acc.clear()
            else:
                in_code = False
                code = "\n".join(code_acc).strip()
                if code:
                    elements.append(RawElement("code", page.page, code, section_path()))
            i += 1
            continue
        if in_code:
            code_acc.append(line)
            i += 1
            continue

        # headings
        hm = HEADING_RE.match(line)
        if hm:
            flush_text(text_acc, "text")
            flush_text(table_acc, "table")
            flush_text(list_acc, "list")
            level, title = len(hm.group(1)), _strip_literal_heading(hm.group(2))
            heading_stack = heading_stack[: level - 1] + [title]
            kind = "title" if level == 1 else "text"
            elements.append(RawElement(kind, page.page, title, section_path()))
            i += 1
            continue

        # tables: consecutive |-lines
        if TABLE_ROW_RE.match(line):
            flush_text(text_acc, "text")
            flush_text(list_acc, "list")
            table_acc.append(line)
            i += 1
            continue
        if table_acc:
            # a line no longer part of the table closes it;
            # fall through so the current line is classified normally
            flush_text(table_acc, "table")

        # lists
        if LIST_ITEM_RE.match(line):
            flush_text(text_acc, "text")
            flush_text(table_acc, "table")
            list_acc.append(line)
            i += 1
            continue
        if list_acc and not LIST_ITEM_RE.match(line) and line.strip():
            flush_text(list_acc, "list")

        # images (standalone or with caption)
        img_match = IMAGE_REF_RE.search(line)
        if img_match:
            flush_text(text_acc, "text")
            flush_text(table_acc, "table")
            flush_text(list_acc, "list")
            ref = img_match.group(1)
            if ref.startswith("http") or ref.startswith("data:"):
                i += 1
                continue
            elements.append(RawElement("image", page.page, ref, section_path()))
            i += 1
            continue

        # equations (block $$...$$)
        if EQUATION_RE.match(line.strip()):
            flush_text(text_acc, "text")
            eq_lines = []
            i += 1
            while i < len(lines) and not EQUATION_RE.match(lines[i].strip()):
                eq_lines.append(lines[i])
                i += 1
            i += 1  # skip closing $$
            eq = "\n".join(eq_lines).strip()
            if eq:
                elements.append(RawElement("equation", page.page, eq, section_path()))
            continue

        # blank line -> end of paragraph
        if not line.strip():
            flush_text(text_acc, "text")
            i += 1
            continue

        # plain text line
        text_acc.append(line)
        i += 1

    # flush trailing accumulators
    flush_text(text_acc, "text")
    flush_text(table_acc, "table")
    flush_text(code_acc, "code" if code_acc else "text")
    flush_text(list_acc, "list")

    return elements


def parse_pages(pages: list[PageMarkdown]) -> list[RawElement]:
    all_elems: list[RawElement] = []
    for page in pages:
        all_elems.extend(parse_page(page))
    return all_elems