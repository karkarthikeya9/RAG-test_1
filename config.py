"""
config.py — shared configuration for the RAG system.

All paths and parameters for ingestion + querying live here so that
component swaps (embedding model, vector DB, markdown engine) are single-line
changes rather than edits scattered across files.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- dirs
DATA_DIR       = "data"        # source PDFs
MARKDOWN_DIR   = "markdown"    # intermediate PDF -> Markdown output
IMAGES_DIR     = "images"      # extracted image files
CHROMA_DIR     = "chroma_db"   # persistent vector store
COLLECTION     = "documents"   # collection name (separate from conversation memory)

# ---------------------------------------------------------------- extraction
# Markdown engine: 'marker' (heavy, best fidelity) — extractor is swappable
MARKDOWN_ENGINE = "marker"

# Marker runs its ML models on CPU by default here (no CUDA guaranteed on win32)
TORCH_DEVICE = "cpu"

# ---------------------------------------------------------------- chunking
CHUNK_SIZE     = 1000   # chars; long text sections are sub-split to this size
CHUNK_OVERLAP  = 200    # overlap between sub-chunks of a long section

# ---------------------------------------------------------------- embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384
EMBEDDING_INSTRUCTION = "Represent this document text for semantic search"

# ---------------------------------------------------------------- llm (groq)
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL      = "openai/gpt-oss-20b"          # answer generation
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"          # image description / flowchart extraction
LLM_TEMPERATURE = 0

# ---------------------------------------------------------------- query
TOP_K        = 5               # knowledge units retrieved per question
MEMORY_TURNS = 5               # hot-memory turns injected into the prompt
LONG_TERM_MEMORY_RESULTS = 3   # relevant past Q&A retrieved

# ---------------------------------------------------------------- vision
IMAGE_QUALITY = 0.9            # PNG/JPEG save quality for extracted images