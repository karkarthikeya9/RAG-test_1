"""
pipeline/store.py — persist KnowledgeUnits into ChromaDB.

Storage vs retrieval separation handled at the record level:
    collection.add(
        ids=...,              unit.id
        documents=...,        unit.content        (storage representation)
        embeddings=...,       embed(embedding_text) (retrieval representation)
        metadatas=...,        type/page/source/section_path/image_path
    )

The stored `documents` string is NOT what vectors come from — the vectors come
from `embeddings` (computed from unit.embedding_text). So we keep the original
structure AND search over the retrieval-friendly representation.
"""

from __future__ import annotations

import os
import shutil

import chromadb

import config
from knowledge_unit import KnowledgeUnit
from pipeline.embeddings import LocalEmbedder


def get_client(wipe: bool = False) -> chromadb.PersistentClient:
    if wipe and os.path.exists(config.CHROMA_DIR):
        shutil.rmtree(config.CHROMA_DIR)
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def get_collection(client: chromadb.PersistentClient, wipe: bool = False):
    try:
        col = client.get_or_create_collection(
            name=config.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        if wipe:
            client.delete_collection(config.COLLECTION)
            col = client.create_collection(
                name=config.COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
    except Exception:
        col = client.create_collection(
            name=config.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return col


def index_units(collection, units: list[KnowledgeUnit], embedder: LocalEmbedder):
    if not units:
        return 0
    ids, docs, metas = [], [], []
    for u in units:
        ids.append(u.id)
        docs.append(u.content)
        metas.append({
            "type": u.type.value,
            "page": u.page,
            "source_document": u.source_document,
            "section_path": u.section_path or ["(root)"],
            "image_path": u.image_path or "",
        })
    vectors = embedder.embed_documents([u.embedding_text for u in units])
    collection.add(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
    return len(ids)


def retrieve(
    collection,
    embedder: LocalEmbedder,
    query: str,
    k: int = config.TOP_K,
    type_filter: str | None = None,
) -> list[dict]:
    where = {"type": type_filter} if type_filter else None
    qv = embedder.embed_query(query)
    count = collection.count()
    if count == 0:
        return []
    n = min(k, count)
    res = collection.query(
        query_embeddings=[qv],
        n_results=n,
        where=where,
    )
    out = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        out.append({
            "id": res["ids"][0][i],
            "content": res["documents"][0][i],
            "embedding_text": "",  # not stored; we reconstruct via metadata below
            "type": meta.get("type"),
            "page": meta.get("page"),
            "source_document": meta.get("source_document"),
            "section_path": meta.get("section_path"),
            "image_path": meta.get("image_path"),
            "distance": res["distances"][0][i],
            "structured_data": None,
        })
    return out