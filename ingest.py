"""
Build the local vector store for the J Street knowledge assistant.

Reads every .txt/.pdf file in jstreet_corpus/, splits each into overlapping
paragraph-based chunks, and loads them into a local Chroma collection.
Embeddings are computed locally by Chroma's default sentence-transformer
model (all-MiniLM-L6-v2, downloaded automatically on first run) — no
embedding API key needed.

Run this once after adding or editing corpus files:
    python ingest.py
"""
import os
import re

import chromadb
from pypdf import PdfReader

CORPUS_DIR = "jstreet_corpus"
DB_DIR = "chroma_db"
COLLECTION_NAME = "jstreet"

# Rough character budget per chunk, with a bit of overlap so a fact that
# falls near a paragraph boundary still shows up whole in one chunk.
CHUNK_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path):
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text):
    # Split on blank lines first so we never cut a sentence in half, then
    # regroup paragraphs into ~CHUNK_CHARS chunks.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) > CHUNK_CHARS:
            chunks.append(current.strip())
            # carry the tail of the previous chunk forward for continuity
            current = current[-CHUNK_OVERLAP_CHARS:] + "\n\n" + para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_index(client):
    """(Re)build the collection from jstreet_corpus/ and return it.

    Also called from app.py on cold start, so a hosted deployment doesn't
    need a separate ingest step — the first request builds the index.
    """
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, documents, metadatas = [], [], []
    corpus_files = sorted(os.listdir(CORPUS_DIR))

    for filename in corpus_files:
        path = os.path.join(CORPUS_DIR, filename)
        lower = filename.lower()
        if lower.endswith(".pdf"):
            text = load_pdf(path)
        elif lower.endswith(".txt"):
            text = load_txt(path)
        else:
            continue

        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"{filename}-{i}")
            documents.append(chunk)
            metadatas.append({"source": filename})

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection, len(ids), len(corpus_files)


def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    _, n_chunks, n_files = build_index(client)
    if n_chunks == 0:
        print("No chunks found — check that jstreet_corpus/ has .txt or .pdf files.")
    else:
        print(f"Indexed {n_chunks} chunks from {n_files} files into {DB_DIR}/")


if __name__ == "__main__":
    main()
