import os
import json
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.genai.errors import ClientError

load_dotenv()

BATCH_SIZE = 5              # chunks embedded per API call
DELAY_BETWEEN_BATCHES = 3   # seconds, stays safely within free-tier rate limits
MAX_RETRIES = 5
REQUEST_TIMEOUT = 30        # seconds — if a call hangs longer than this, treat as failed and retry

_executor = ThreadPoolExecutor(max_workers=1)


def add_documents_with_timeout(store, batch, timeout=REQUEST_TIMEOUT):
    """Run store.add_documents in a worker thread with a hard timeout, so a
    silently-hanging network call (e.g. a stalling corporate proxy) fails
    fast instead of freezing the script forever."""
    future = _executor.submit(store.add_documents, batch)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        raise TimeoutError(f"Embedding call did not respond within {timeout}s (likely a network stall)")

INPUT_FILE = "chunks.json"           # built by extract_text.py + chunk_text.py (unchanged)
STORE_PATH = "vector_store.json"     # LangChain's InMemoryVectorStore persisted as JSON

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "No API key found. Add this to your .env file:\n"
        "  GOOGLE_API_KEY=your_key_here"
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
    task_type="retrieval_document"
)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_or_create_store():
    if os.path.exists(STORE_PATH):
        print(f"Loading existing vector store from {STORE_PATH}...")
        return InMemoryVectorStore.load(STORE_PATH, embeddings)
    print("No existing vector store found — creating a new one.")
    return InMemoryVectorStore(embeddings)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"'{INPUT_FILE}' not found. Run extract_text.py + chunk_text.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    store = load_or_create_store()

    # Figure out what's already embedded (by chunk id) and its content hash,
    # so we only re-embed chunks that are new or changed.
    existing_hashes = {}
    for doc_id, doc in store.store.items():
        meta = doc.get("metadata", {})
        chunk_id = meta.get("chunk_id")
        if chunk_id is not None:
            existing_hashes[chunk_id] = meta.get("text_hash")

    current_chunk_ids = {str(c["id"]) for c in chunks}

    # Drop entries for chunks that no longer exist (source doc removed/changed)
    stale_doc_ids = [
        doc_id for doc_id, doc in store.store.items()
        if doc.get("metadata", {}).get("chunk_id") not in current_chunk_ids
    ]
    if stale_doc_ids:
        store.delete(ids=stale_doc_ids)
        print(f"Removed {len(stale_doc_ids)} stale chunks no longer in source docs.")

    to_embed = []
    for chunk in chunks:
        chunk_id = str(chunk["id"])
        h = text_hash(chunk["text"])
        if existing_hashes.get(chunk_id) == h:
            continue  # unchanged, skip re-embedding
        to_embed.append((chunk, h))

    if not to_embed:
        print("Nothing new or changed — vector store is already up to date.")
        return

    print(f"Embedding {len(to_embed)} new/changed chunks in batches of {BATCH_SIZE} "
          f"(skipped {len(chunks) - len(to_embed)} unchanged)...")

    all_docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": str(chunk["id"]),
                "source": chunk["source"],
                "section_path": chunk.get("section_path", ""),
                "chunk_type": chunk.get("chunk_type", "text"),
                "text_hash": h
            }
        )
        for chunk, h in to_embed
    ]

    total_batches = (len(all_docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        batch = all_docs[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]

        attempt = 0
        while True:
            try:
                add_documents_with_timeout(store, batch)
                break
            except (ClientError, TimeoutError, Exception) as e:
                is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                is_timeout = isinstance(e, TimeoutError)
                attempt += 1
                if (is_rate_limit or is_timeout) and attempt <= MAX_RETRIES:
                    wait = min(60, 2 ** attempt * 5)  # exponential backoff, capped at 60s
                    reason = "Timed out" if is_timeout else "Rate limited"
                    print(f"  {reason}. Retrying batch {batch_num+1}/{total_batches} "
                          f"in {wait}s (attempt {attempt}/{MAX_RETRIES})...")
                    time.sleep(wait)
                    continue
                else:
                    # Save whatever we've embedded so far before giving up,
                    # so progress is never lost on a hard failure.
                    store.dump(STORE_PATH)
                    print(f"\nStopped at batch {batch_num+1}/{total_batches} due to error: {e}")
                    print(f"Progress saved to '{STORE_PATH}' ({len(store.store)} chunks so far). "
                          f"Just re-run this script later to resume from here.")
                    return

        # Save after every successful batch, so a crash/rate-limit never
        # loses more than one batch's worth of work.
        store.dump(STORE_PATH)
        print(f"[{batch_num+1}/{total_batches}] embedded {len(batch)} chunks "
              f"(saved, {len(store.store)} total so far)")

        if batch_num < total_batches - 1:
            time.sleep(DELAY_BETWEEN_BATCHES)

    print(f"\nDone. Vector store saved to '{STORE_PATH}' "
          f"({len(store.store)} total chunks).")


if __name__ == "__main__":
    main()