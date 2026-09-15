"""
chat.py — Pipeline B: Questioning / Retrieval (with multi-turn memory)

    Question → Retrieve (Chroma) + Memory → Prompt → Groq LLM → Answer

Requires the index built by ingest.py (run it first):

    python ingest.py
    python chat.py

Uses the SAME local embedding model as ingest.py so the vectors match.
"""

import os
import re
import sys
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import chromadb

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# ---------------------------------------------------------------- config
CHROMA_DIR      = "chroma_db"     # must match ingest.py
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # must match ingest.py
GROQ_MODEL      = "openai/gpt-oss-20b"
TOP_K           = 4               # how many document chunks to retrieve per question
HOT_MEMORY_TURNS = 10            # number of recent turns to keep in hot memory
LONG_TERM_MEMORY_RESULTS = 3     # number of relevant past Q&A to retrieve

SYSTEM_PROMPT = """You are a helpful, friendly study assistant. You can answer questions about the user's
course notes AND have general knowledge about any topic.

Context contains two types of labeled sources:
- [M1], [M2], [M3] ... are past conversation chunks (from previous Q&A in this session)
- [1], [2], [3] ... are document chunks (from course notes PDF)

Rules:
- When answering from labeled context (either [M1] or [1] style), cite them inline like [M1] or [1].
- When answering from your own knowledge (greetings, general questions, topics not in the context), do NOT cite any sources.
- Be conversational and warm. Greet users back when they say hi/hello/thanks.
- Always be helpful. If the context doesn't help, answer from your own knowledge.
- After your answer, add a line: SOURCES_USED: [M1], [2] (or SOURCES_USED: none if you used your own knowledge)
"""


# ---------------------------------------------------------------- hot memory
@dataclass
class ConversationTurn:
    turn_number: int
    question: str
    answer: str
    sources_used: list[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class HotMemory:
    """Keeps last N turns in a fast deque for immediate context."""

    def __init__(self, max_turns: int = HOT_MEMORY_TURNS):
        self.max_turns = max_turns
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    def add_turn(self, question: str, answer: str, sources: list[int]):
        turn = ConversationTurn(
            turn_number=len(self.turns) + 1,
            question=question,
            answer=answer,
            sources_used=sources,
        )
        self.turns.append(turn)

    def get_recent_messages(self, n: int = 5) -> list[dict]:
        """Return last N turns as LLM message dicts."""
        recent = list(self.turns)[-n:]
        messages = []
        for turn in recent:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def clear(self):
        self.turns.clear()


# ---------------------------------------------------------------- long-term memory
class LongTermMemory:
    """Stores every Q&A pair as vectors in an in-memory ChromaDB."""

    def __init__(self, embedder: HuggingFaceEmbeddings):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder
        self.conversation_id = str(uuid.uuid4())
        self.turn_count = 0

    def store_qa(self, question: str, answer: str):
        self.turn_count += 1
        embedding = self.embedder.embed_query(question)
        self.collection.add(
            ids=[f"{self.conversation_id}_turn_{self.turn_count}"],
            embeddings=[embedding],
            documents=[f"Q: {question}\nA: {answer}"],
            metadatas=[{
                "conversation_id": self.conversation_id,
                "turn_number": self.turn_count,
                "timestamp": datetime.now().isoformat(),
            }],
        )

    def retrieve_relevant(self, query: str, n_results: int = LONG_TERM_MEMORY_RESULTS) -> list[dict]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedder.embed_query(query)
        count = min(n_results, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=count,
        )
        output = []
        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            lines = doc.split("\n")
            q = lines[0].replace("Q: ", "") if lines else ""
            a = lines[1].replace("A: ", "") if len(lines) > 1 else ""
            output.append({
                "question": q,
                "answer": a,
                "distance": dist,
                "turn_number": meta.get("turn_number"),
                "timestamp": meta.get("timestamp"),
            })
        return output

    def clear(self):
        self.client.delete_collection("conversation_memory")
        self.collection = self.client.create_collection(
            name="conversation_memory",
            metadata={"hnsw:space": "cosine"},
        )
        self.conversation_id = str(uuid.uuid4())
        self.turn_count = 0


# ---------------------------------------------------------------- source extraction
def extract_sources(answer: str) -> tuple[str, list[str]]:
    """Pull the SOURCES_USED line out of the LLM response.
    Returns (clean_answer, sources) where sources are tokens like "1", "M1", "M2".
    """
    match = re.search(r"SOURCES_USED:\s*(.+)", answer, re.IGNORECASE)
    sources: list[str] = []
    if match:
        tag = match.group(1).strip().lower()
        if tag != "none":
            sources = re.findall(r"[m]?\d+", tag)
            sources = [s.upper() if s.startswith("m") else s for s in sources]
        clean = answer[: match.start()].rstrip()
    else:
        clean = answer
    return clean, sources


# ---------------------------------------------------------------- document retrieval
def retrieve_docs(db: Chroma, question: str, k: int = TOP_K):
    return db.similarity_search(question, k=k)


def format_doc_context(docs) -> str:
    return "\n\n".join(
        f"[{i + 1}] {doc.page_content}\n(source: {doc.metadata.get('source_document')}, "
        f"page {doc.metadata.get('page')})"
        for i, doc in enumerate(docs)
    )


# ---------------------------------------------------------------- main loop
def main():
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("Missing GROQ_API_KEY — copy .env.example to .env and fill it in.")

    if not os.path.exists(CHROMA_DIR):
        raise SystemExit(f"No {CHROMA_DIR}/ found — run 'python ingest.py' first.")

    print(f"[EMBED]     loading {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    doc_db = Chroma(
        collection_name="documents",
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    print(f"[LLM]       {GROQ_MODEL}")
    print(f"[MEMORY]    hot={HOT_MEMORY_TURNS} turns, long-term retrieval=top {LONG_TERM_MEMORY_RESULTS}\n")

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    hot_memory = HotMemory(max_turns=HOT_MEMORY_TURNS)
    long_term_memory = LongTermMemory(embedder=embeddings)

    print("Ask a question about your notes (commands: 'quit', 'clear').\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            break
        if question.lower() == "clear":
            hot_memory.clear()
            long_term_memory.clear()
            print(">> Memory cleared.\n")
            continue

        # ---- 1. retrieve relevant past Q&A from long-term memory
        past_qa = long_term_memory.retrieve_relevant(question)

        # ---- 2. retrieve document chunks
        doc_chunks = retrieve_docs(doc_db, question)

        # ---- 3. build context block
        context_parts: list[str] = []

        if past_qa:
            context_parts.append("Relevant past conversations:")
            for i, p in enumerate(past_qa, start=1):
                context_parts.append(
                    f"[M{i}] Q: {p['question']}\n"
                    f"      A: {p['answer']}\n"
                    f"      (turn {p['turn_number']}, {p['timestamp']})"
                )

        if doc_chunks:
            context_parts.append("\nCourse notes:")
            context_parts.append(format_doc_context(doc_chunks))

        full_context = "\n\n".join(context_parts) if context_parts else "No relevant context found."

        # ---- 4. assemble messages
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"Context:\n{full_context}"},
        ]

        # add recent conversation history from hot memory
        messages.extend(hot_memory.get_recent_messages(n=5))

        # add the current question
        messages.append({"role": "user", "content": question})

        # ---- 5. call Groq directly
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0,
        )
        raw_answer = response.choices[0].message.content
        answer, sources_used = extract_sources(raw_answer)

        # ---- 6. display
        print(f"\nAssistant: {answer}\n")

        if sources_used:
            mem_sources = [s for s in sources_used if s.upper().startswith("M")]
            doc_sources = [s for s in sources_used if not s.upper().startswith("M")]
            printed_any = False

            if mem_sources:
                print("Past conversation sources:")
                for token in mem_sources:
                    idx = int(token[1:])
                    if 1 <= idx <= len(past_qa):
                        p = past_qa[idx - 1]
                        print(f"  [{token}] previous Q: {p['question']}")
                        print(f"         previous A: {p['answer']}")
                        print(f"         turn {p['turn_number']}, at {p['timestamp']} ")
                        printed_any = True

            if doc_sources:
                print("Document sources:")
                for token in doc_sources:
                    idx = int(token)
                    if 1 <= idx <= len(doc_chunks):
                        doc = doc_chunks[idx - 1]
                        src = os.path.basename(doc.metadata.get("source_document", "?"))
                        page = doc.metadata.get("page", "?")
                        print(f"  [{idx}] {src}, page {page}")
                        printed_any = True

            if printed_any:
                print()

        # ---- 7. store in both memory layers
        hot_memory.add_turn(question, answer, sources_used)
        long_term_memory.store_qa(question, answer)

    # cleanup on exit
    hot_memory.clear()
    long_term_memory.clear()
    print("Session ended. Memory cleared.")


if __name__ == "__main__":
    main()
