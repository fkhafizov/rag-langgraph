"""
build_chromadb_rag_papers.py

Downloads five foundational RAG papers from arXiv, chunks and embeds them
with the OpenAI API, and stores everything in a local persistent ChromaDB
collection for later Q&A.

Environment: conda env `rag_openai_local` (needs chromadb added — see
requirements note below).

Usage:
    python build_chromadb_rag_papers.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chromadb
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200       # characters
CHUNK_OVERLAP = 200     # characters

PDF_DIR = Path("papers")
CHROMA_DIR = "chroma_rag_papers"
COLLECTION_NAME = "rag_foundational_papers"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Make sure it's in your .env file or shell environment."
    )
client = OpenAI(api_key=api_key)

# --------------------------------------------------------------------------
# The five papers
# --------------------------------------------------------------------------

PAPERS = [
    {
        "id": "lewis2020_rag",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al., 2020",
        "abs_url": "https://arxiv.org/abs/2005.11401",
    },
    {
        "id": "karpukhin2020_dpr",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Karpukhin et al., 2020",
        "abs_url": "https://arxiv.org/abs/2004.04906",
    },
    {
        "id": "gao2023_rag_survey",
        "title": "Retrieval-Augmented Generation for Large Language Models: A Survey",
        "authors": "2023",
        "abs_url": "https://arxiv.org/abs/2312.10997",
    },
    {
        "id": "cheng2025_knowledge_rag_survey",
        "title": "A Survey on Knowledge-Oriented Retrieval-Augmented Generation",
        "authors": "Cheng et al.",
        "abs_url": "https://arxiv.org/abs/2503.10677",
    },
    {
        "id": "brown2025_rag_slr",
        "title": "A Systematic Literature Review of Retrieval-Augmented Generation: "
                 "Techniques, Metrics, and Challenges",
        "authors": "Brown, Roman & Devereux",
        "abs_url": "https://arxiv.org/abs/2508.06401",
    },
]


def abs_to_pdf_url(abs_url: str) -> str:
    """Convert an arxiv.org/abs/XXXX.XXXXX URL to its PDF equivalent."""
    arxiv_id = abs_url.rstrip("/").split("/")[-1]
    return f"https://arxiv.org/pdf/{arxiv_id}"


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download_pdf(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Downloaded: {dest}")
    return dest


# --------------------------------------------------------------------------
# Extract + chunk
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: str
    text: str
    paper_id: str
    title: str
    authors: str
    source_url: str


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append(f"\n[Page {page_number}]\n{text}")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    clean_text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))

        if end < len(clean_text):
            sentence_end = clean_text.rfind(". ", start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break
        start = end - overlap

    return chunks


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------

def create_embeddings(texts: List[str], model: str = EMBEDDING_MODEL, batch_size: int = 100) -> List[List[float]]:
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    all_chunks: List[Chunk] = []

    for paper in PAPERS:
        pdf_url = abs_to_pdf_url(paper["abs_url"])
        pdf_path = PDF_DIR / f"{paper['id']}.pdf"

        download_pdf(pdf_url, pdf_path)

        text = extract_text(pdf_path)
        print(f"  Characters extracted: {len(text)}")

        pieces = chunk_text(text)
        print(f"  Chunks: {len(pieces)}")

        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    chunk_id=f"{paper['id']}_chunk{i}",
                    text=piece,
                    paper_id=paper["id"],
                    title=paper["title"],
                    authors=paper["authors"],
                    source_url=paper["abs_url"],
                )
            )

    print(f"\nTotal chunks across all papers: {len(all_chunks)}")

    print("Creating embeddings (this calls the OpenAI API and incurs cost)...")
    embeddings = create_embeddings([c.text for c in all_chunks])
    print(f"Embedding matrix: {len(embeddings)} vectors of dimension {len(embeddings[0])}")

    # ----------------------------------------------------------------------
    # Store in ChromaDB (persistent, local)
    # ----------------------------------------------------------------------

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Fresh collection each run — drop if it already exists
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Five foundational RAG papers for Q&A"},
    )

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[
            {
                "paper_id": c.paper_id,
                "title": c.title,
                "authors": c.authors,
                "source_url": c.source_url,
            }
            for c in all_chunks
        ],
    )

    print(f"\nSaved to ChromaDB at ./{CHROMA_DIR} (collection: {COLLECTION_NAME})")
    print(f"Total vectors stored: {collection.count()}")


if __name__ == "__main__":
    main()
