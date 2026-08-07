"""
rag_openai_local.py

Local (macOS) port of rag-openai-v2.ipynb.

Ported from the notebook's manual pipeline (chunk -> embed -> cosine
similarity retrieval -> grounded generation). The LlamaIndex
VectorStoreIndex path from the notebook is dropped here since the manual
pipeline is the one that was actually built out with save/reload support.

Usage:
    python rag_openai_local.py --file paper.pdf --ask "What is the main argument?"
    python rag_openai_local.py --url https://arxiv.org/pdf/2007.01282.pdf --ask "Summarize the contributions."
    python rag_openai_local.py --reload --ask "What is the main argument?"   # skip re-embedding
"""


"""
 $   python rag_openai_local.py --file ../Five_Foundational_RAG_Papers-claude.pdf --ask "What is the main argument?"
Characters: 2228
Number of chunks: 3
Embedding matrix shape: (3, 1536)
Saved: rag_chunks.json, rag_embeddings.npy

QUESTION: What is the main argument?
ANSWER: Document statement (supported by the excerpts):
- The document's purpose is to present foundational papers and recent surveys on Retrieval-Augmented Generation (RAG) as a curated reference to guide RAG/vector-database work — i.e., it collects the original foundational works and systematic surveys that catalogue datasets, architectures, evaluation practices, and the progression of RAG methods. Support: "Five Foundational RAG Papers Reference list for building a RAG vector database application" [Chunk 0]; and the description of the systematic review "cataloguing datasets, architectures, and evaluation practices — useful as a curated map of the field." [Chunk 2]; and the survey framing the progression "from Naive RAG to Advanced RAG to Modular RAG" and examining core components and evaluation frameworks [Chunk 1].

My inference:
- The main argument implied by the document is that these foundational papers and surveys together form a curated map and essential reference set for understanding and designing RAG systems and vector databases (in other words, the field should be approached via these key works and synthesized surveys). This inference follows from the cited statements that the list is a "reference list for building a RAG vector database application" and that the reviews "catalogue" and "examine" the field [Chunk 0; Chunk 2; Chunk 1].

Retrieved chunks:
 rank  chunk_id    score
    1         2 0.161249
    2         1 0.153704
    3         0 0.143780

"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

load_dotenv()  # reads OPENAI_API_KEY from a local .env file

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-5-mini"

CHUNK_SIZE = 1200       # characters
CHUNK_OVERLAP = 200     # characters
TOP_K = 5

CHUNKS_PATH = "rag_chunks.json"
EMBEDDINGS_PATH = "rag_embeddings.npy"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Create a .env file (see .env.example) "
        "or export it in your shell."
    )
client = OpenAI(api_key=api_key)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def download_pdf(url: str, dest: str = "paper.pdf") -> str:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest, "wb") as f:
        f.write(response.content)
    print("Downloaded:", dest)
    return dest


def load_document(file_path: str) -> str:
    path = Path(file_path)

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(f"\n[Page {page_number}]\n{text}")
        return "\n".join(pages)

    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    raise ValueError(
        f"Unsupported file type: {path.suffix}. Please provide a PDF, TXT, or Markdown file."
    )


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    clean_text = " ".join(text.split())
    chunks = []
    start = 0
    chunk_id = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))

        if end < len(clean_text):
            sentence_end = clean_text.rfind(". ", start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk))
            chunk_id += 1

        if end >= len(clean_text):
            break
        start = end - overlap

    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def create_embeddings(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100) -> np.ndarray:
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return np.asarray(all_embeddings, dtype=np.float32)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def embed_query(query: str) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    return vector / max(np.linalg.norm(vector), 1e-12)


# --------------------------------------------------------------------------
# Save / reload index
# --------------------------------------------------------------------------

def save_index(chunks: List[Chunk], embeddings: np.ndarray) -> None:
    index_df = pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "text": [c.text for c in chunks],
        }
    )
    index_df.to_json(CHUNKS_PATH, orient="records", force_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved: {CHUNKS_PATH}, {EMBEDDINGS_PATH}")


def load_index() -> tuple[List[Chunk], np.ndarray]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        saved_chunks = json.load(f)
    chunks = [Chunk(chunk_id=item["chunk_id"], text=item["text"]) for item in saved_chunks]
    embeddings = np.load(EMBEDDINGS_PATH)
    print("Loaded chunks:", len(chunks))
    print("Embedding matrix:", embeddings.shape)
    return chunks, embeddings


# --------------------------------------------------------------------------
# Retrieval + generation
# --------------------------------------------------------------------------

def retrieve(query: str, chunks: List[Chunk], normalized_embeddings: np.ndarray, top_k: int = TOP_K) -> pd.DataFrame:
    query_embedding = embed_query(query)
    scores = normalized_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(
            {
                "rank": rank,
                "chunk_id": chunks[idx].chunk_id,
                "score": float(scores[idx]),
                "text": chunks[idx].text,
            }
        )
    return pd.DataFrame(results)


def answer_question(question: str, chunks: List[Chunk], normalized_embeddings: np.ndarray, top_k: int = TOP_K) -> dict:
    retrieved = retrieve(question, chunks, normalized_embeddings, top_k=top_k)

    context_parts = [f"[Chunk {row.chunk_id}]\n{row.text}" for row in retrieved.itertuples()]
    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the supplied context.

Rules:
1. Do not use outside knowledge.
2. If the context is insufficient, say that the document does not provide
   enough information.
3. Cite supporting passages using chunk identifiers such as [Chunk 4].
4. Distinguish clearly between statements from the document and any
   inference you make.

Question:
{question}

Context:
{context}
""".strip()

    response = client.responses.create(model=GENERATION_MODEL, input=prompt)

    return {"question": question, "answer": response.output_text, "retrieved": retrieved}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Local RAG over a PDF/TXT/MD document.")
    parser.add_argument("--file", type=str, help="Path to a local PDF/TXT/MD file.")
    parser.add_argument("--url", type=str, help="URL of a PDF to download and index.")
    parser.add_argument("--reload", action="store_true", help="Reload a previously saved index instead of re-embedding.")
    parser.add_argument("--ask", type=str, action="append", required=True, help="Question to ask (repeatable).")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    if args.reload:
        chunks, embeddings = load_index()
    else:
        if args.url:
            file_path = download_pdf(args.url)
        elif args.file:
            file_path = args.file
        else:
            raise SystemExit("Provide --file, --url, or --reload.")

        document_text = load_document(file_path)
        print("Characters:", len(document_text))

        chunks = chunk_text(document_text)
        print("Number of chunks:", len(chunks))

        embeddings = create_embeddings([c.text for c in chunks])
        print("Embedding matrix shape:", embeddings.shape)

        save_index(chunks, embeddings)

    normalized_embeddings = normalize_rows(embeddings)

    for question in args.ask:
        result = answer_question(question, chunks, normalized_embeddings, top_k=args.top_k)
        print("\nQUESTION:", result["question"])
        print("ANSWER:", result["answer"])
        print("\nRetrieved chunks:")
        print(result["retrieved"][["rank", "chunk_id", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
