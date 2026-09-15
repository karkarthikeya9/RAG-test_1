"""
knowledge_unit.py — the core data model.

A KnowledgeUnit represents one retrievable element extracted from a document.

Key distinction (per the architecture spec):
    content        = storage representation (what we preserve from the source)
    embedding_text = retrieval representation (what we embed / search)

For example an image has:
    content          = a reference to the saved PNG on disk
    embedding_text   = the VLM-generated retrieval-oriented description
    structured_data  = flowchart algorithm (if any)
    image_path       = original file location

And a table has:
    content          = the markdown table
    structured_data  = {headers, rows}
    embedding_text   = prose form ("200 means OK ...")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class KnowledgeUnitType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FLOWCHART = "flowchart"
    DIAGRAM = "diagram"
    CODE = "code"
    EQUATION = "equation"
    LIST = "list"
    TITLE = "title"


@dataclass
class KnowledgeUnit:
    type: KnowledgeUnitType
    content: str                                # storage representation
    embedding_text: str                         # retrieval representation
    page: int = 0
    source_document: str = ""
    section_path: list[str] = field(default_factory=list)
    image_path: str | None = None               # original image (image/flowchart/diagram)
    structured_data: dict | None = None         # table headers/rows, flowchart algorithm...
    description: str | None = None              # VLM description
    id: str = ""

    def __post_init__(self):
        if not self.id:
            h = hashlib.md5(
                f"{self.source_document}|{self.page}|{self.type}|{self.embedding_text[:60]}".encode()
            ).hexdigest()[:12]
            self.id = f"ku_{h}"

    def to_chroma_record(self):
        """Return (id, document, embedding_text, metadata) for the vector store."""
        meta = {
            "type": self.type.value,
            "page": self.page,
            "source_document": self.source_document,
            "section_path": self.section_path or ["(root)"],
            "image_path": self.image_path or "",
        }
        # metadata must be flat & primitives; section_path is a list so keep it too (chroma accepts lists)
        return self.id, self.content, self.embedding_text, meta

    def short(self, width: int = 80) -> str:
        t = self.embedding_text.replace("\n", " ")
        return t[:width] + ("..." if len(t) > width else "")