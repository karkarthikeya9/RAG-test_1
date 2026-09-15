# simple-rag

A minimal two-file RAG over your FSD notes PDF.

```
simple-rag/
├── data/
│   └── FSD_Complete_Notes_Unit1_&_Unit2.pdf   ← source documents
├── chroma_db/                                  ← created by ingest.py (do not edit)
├── ingest.py    ← Pipeline A: PDF → LOAD → TEXT → CHUNK → EMBED → STORE
├── chat.py      ← Pipeline B: Question → Retrieve → Prompt → LLM → Answer
├── requirements.txt
├── .env          ← your OPENAI_API_KEY (never commit this)
└── README.md
```

## Setup

```bash
cd simple-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # then paste your OpenAI key into .env
```

## Run

```bash
# Step 1 — build the index (run again whenever you add/update PDFs in data/)
python ingest.py

# Step 2 — ask questions (once chat.py exists)
python chat.py
```

## How Pipeline A works (ingest.py)

| Step | What happens | Code |
|------|--------------|------|
| LOAD | PyMuPDF opens the PDF, one Document per page | `PyMuPDFLoader` |
| TEXT | Page text becomes `page_content` + metadata (page number, source file) | `Document` objects |
| CHUNK | Pages are long, so a `RecursiveCharacterTextSplitter` cuts them into ~1000-char overlapping pieces | `split_documents()` |
| EMBED | Each chunk is sent to `text-embedding-3-small`, producing a 1536-dim vector | `OpenAIEmbeddings` |
| STORE | ChromaDB saves vectors + original text to `chroma_db/` on disk | `Chroma.from_documents()` |

After this runs, `chroma_db/` contains everything retrieval needs — no PDF
parsing or API calls happen at question time (except the LLM itself).
