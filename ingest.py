"""
ingest.py
---------
Loads company policy documents (PDF / DOCX) from a folder, splits them into
chunks, embeds them locally, and stores them in a local Chroma vector DB.

Usage:
    python ingest.py --docs_dir ./policies --db_dir ./chroma_db
"""

import argparse
import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_dir: str):
    """Load all PDF and DOCX files from a directory, with OCR fallback for scanned PDFs."""
    docs = []
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    for file_path in docs_path.rglob("*"):
        if file_path.suffix.lower() == ".pdf":
            try:
                loader = PyPDFLoader(str(file_path))
                loaded = loader.load()
                # If PyPDFLoader extracted almost no text, the PDF is likely scanned -> OCR fallback
                total_chars = sum(len(d.page_content.strip()) for d in loaded)
                if total_chars < 50:
                    print(f"  -> Low text yield, retrying with OCR: {file_path.name}")
                    loader = UnstructuredPDFLoader(str(file_path), strategy="ocr_only")
                    loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded PDF: {file_path.name} ({len(loaded)} pages)")
            except Exception as e:
                print(f"  !! Failed to load {file_path.name}: {e}")

        elif file_path.suffix.lower() in (".docx", ".doc"):
            try:
                loader = Docx2txtLoader(str(file_path))
                loaded = loader.load()
                docs.extend(loaded)
                print(f"Loaded DOCX: {file_path.name}")
            except Exception as e:
                print(f"  !! Failed to load {file_path.name}: {e}")

    # Attach source filename as metadata (used later for citations in the chatbot)
    for d in docs:
        d.metadata["source"] = Path(d.metadata.get("source", "unknown")).name

    return docs


def chunk_documents(docs, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(chunks, db_dir, embedding_model="BAAI/bge-small-en-v1.5"):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir,
        collection_name="company_policies",
    )
    return vectordb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default="./policies", help="Folder with policy PDF/DOCX files")
    parser.add_argument("--db_dir", default="./chroma_db", help="Where to persist the vector DB")
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    print(f"Loading documents from: {args.docs_dir}")
    docs = load_documents(args.docs_dir)
    print(f"Loaded {len(docs)} document pages/sections total.")

    if not docs:
        print("No documents found. Add PDF/DOCX policy files to the docs folder and re-run.")
        return

    print("Chunking documents...")
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"Created {len(chunks)} chunks.")

    print("Embedding and storing in Chroma (this runs locally, may take a moment)...")
    build_vector_store(chunks, args.db_dir)
    print(f"Done. Vector DB saved to: {args.db_dir}")


if __name__ == "__main__":
    main()