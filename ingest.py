import os
import re

import chromadb
from pypdf import PdfReader

CORPUS_DIR = "jstreet_corpus"
DB_DIR = "chroma_db"
COLLECTION_NAME = "jstreet"

# will overlap so the context is not lost
CHUNK_CHARS = 1000
CHUNK_OVERLAP_CHARS = 150


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path):
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text):
    # Split on blank lines first so we never cut a sentence in half, then regroup paragraphs into ~CHUNK_CHARS chunks.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # A "paragraph" bigger than a whole chunk (e.g. a name directory with no
    # blank lines between entries) would otherwise become one oversized chunk
    # whose embedding is diluted across everything in it. Break those up by
    # line first so they still get grouped into normal-sized chunks below.
    pieces = []
    for para in paragraphs:
        if len(para) <= CHUNK_CHARS:
            pieces.append(para)
        else:
            lines = [l.strip() for l in para.split("\n") if l.strip()]
            if len(lines) > 1:
                pieces.extend(lines)
            else:
                pieces.extend(para[i:i + CHUNK_CHARS] for i in range(0, len(para), CHUNK_CHARS))

    chunks = []
    current = ""
    for para in pieces:
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
        print("No chunks found!")
    else:
        print(f"Indexed {n_chunks} chunks from {n_files} files into {DB_DIR}/")


if __name__ == "__main__":
    main()
