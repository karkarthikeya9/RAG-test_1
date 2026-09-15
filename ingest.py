"""
ingest.py — Pipeline A: PDF -> Markdown -> Knowledge Units -> Vector Store

    PDF
     -> marker (PDF -> Markdown, page markers + extracted images)
     -> structural parser (headings/paragraphs/tables/images/code...)
     -> element builder (KnowledgeUnits: text, table, image, flowchart, code...)
     -> vision (Groq Qwen: classify + retrieval-oriented description + flowchart algorithm)
     -> embed ONLY embedding_text -> ChromaDB (storage representation kept in doc)

Run:

    python ingest.py                  # full multimodal ingestion (uses vision)
    python ingest.py --no-vision      # skip VLM calls (text/tables only, faster)

Requires GROQ_API_KEY in .env for the vision step.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from knowledge_unit import KnowledgeUnitType
from pipeline import parser as md_parser
from pipeline import store as db
from pipeline.embeddings import LocalEmbedder
from pipeline.extract_markdown import convert_pdf
from pipeline.elements import build_units_from_elements, resolve_images


def ingest_one_pdf(pdf_path: str, embedder, collection, use_vision: bool) -> int:
    print(f"\n[EXTRACT] {pdf_path}")
    extract = convert_pdf(pdf_path, out_dir=config.MARKDOWN_DIR)
    print(f"[EXTRACT] {len(extract.pages)} pages, "
          f"{len(extract.images)} embedded image(s): {list(extract.images)}")

    # markdown-aware structural parsing
    raw = md_parser.parse_pages(extract.pages)
    print(f"[PARSE]   {len(raw)} raw elements: "
          f"{', '.join(sorted({e.kind for e in raw}))}")

    units = build_units_from_elements(raw, extract.source_document)
    print(f"[BUILD]   {len(units)} knowledge units "
          f"({', '.join(sorted({u.type.value for u in units}))})")

    units = resolve_images(units, extract)
    imgs = [u for u in units if u.type is KnowledgeUnitType.IMAGE and u.image_path]
    if imgs:
        print(f"[IMAGES]  saved: {[u.image_path for u in imgs]}")

    # vision pass (classify + describe + flowchart algorithm)
    from pipeline import vision

    if use_vision:
        for u in units:
            if u.type is KnowledgeUnitType.IMAGE and u.image_path:
                print(f"[VISION]  {u.image_path} ...")
                vision.process_image_unit(u)
    else:
        for u in units:
            if u.type is KnowledgeUnitType.IMAGE and u.image_path:
                u.embedding_text = f"[image on page {u.page}: {u.image_path}]"

    flowchart = [u for u in units if u.type is KnowledgeUnitType.FLOWCHART]
    if flowchart:
        print(f"[FLOW]    {len(flowchart)} flowchart unit(s) with algorithms")

    n = db.index_units(collection, units, embedder)
    print(f"[STORE]   {n} vectors added")
    return n


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG ingestion")
    parser.add_argument("--no-vision", action="store_true",
                        help="skip VLM image description (text/tables only)")
    parser.add_argument("--pdf", type=str, default=None,
                        help="ingest a single PDF file instead of data/*.pdf")
    args = parser.parse_args()

    if not os.path.isdir(config.DATA_DIR):
        raise SystemExit(f"No {config.DATA_DIR}/ directory found.")

    pdfs = [args.pdf] if args.pdf else [
        os.path.join(config.DATA_DIR, f)
        for f in sorted(os.listdir(config.DATA_DIR))
        if f.lower().endswith(".pdf")
    ]
    if not pdfs:
        raise SystemExit(f"No PDFs found in ./{config.DATA_DIR}/")

    if args.pdf and not os.path.exists(args.pdf):
        raise SystemExit(f"PDF not found: {args.pdf}")

    embedder = LocalEmbedder()
    client = db.get_client(wipe=True)
    collection = db.get_collection(client, wipe=True)

    total = 0
    for pdf in pdfs:
        try:
            total += ingest_one_pdf(pdf, embedder, collection, use_vision=not args.no_vision)
        except Exception as exc:
            print(f"[ERROR]   {pdf}: {exc}")

    print(f"\n[DONE] {total} knowledge units indexed -> {config.COLLECTION} "
          f"in {config.CHROMA_DIR}/")
    print("[DONE] Now run: python chat.py")


if __name__ == "__main__":
    main()