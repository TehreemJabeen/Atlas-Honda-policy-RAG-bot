import os
import json

INPUT_FOLDER = "extracted_structure"   # .json files from Step 2
OUTPUT_FILE = "chunks.json"

MAX_CHUNK_WORDS = 350   # soft cap before we split a long section into multiple chunks
MIN_CHUNK_WORDS = 40    # avoid tiny orphan chunks; merge forward if under this


def table_to_text(rows):
    """Render a table as readable pipe-delimited text (keeps it as ONE chunk)."""
    if not rows:
        return ""
    header, *body = rows
    lines = [" | ".join(header)]
    lines.append(" | ".join(["---"] * len(header)))
    for row in body:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def section_path_str(stack):
    """Turn the current heading stack into a readable breadcrumb, e.g.
    'Leave and Time Off Policy > Annual Leave Entitlement'."""
    return " > ".join(h["text"] for h in stack if h["text"])


def flush(buffer_words, stack, source, chunk_id, chunk_type="text"):
    """Package the current buffer into a chunk dict, prefixed with its section path
    so the embedding captures section context, not just raw sentences."""
    if not buffer_words:
        return None
    path = section_path_str(stack)
    body_text = " ".join(buffer_words)
    full_text = f"[{path}]\n{body_text}" if path else body_text
    return {
        "id": chunk_id,
        "source": source,
        "section_path": path,
        "chunk_type": chunk_type,
        "text": full_text
    }


def chunk_document(doc_data):
    source = doc_data["source"]
    elements = doc_data["elements"]

    chunks = []
    chunk_id_counter = [0]

    def next_id():
        cid = f"{source}_{chunk_id_counter[0]}"
        chunk_id_counter[0] += 1
        return cid

    # stack of currently "open" headings, one slot per level (0=title, 1=H1, ...)
    heading_stack = []
    buffer_words = []

    def flush_buffer(chunk_type="text"):
        nonlocal buffer_words
        chunk = flush(buffer_words, heading_stack, source, next_id(), chunk_type)
        if chunk:
            chunks.append(chunk)
        buffer_words = []

    for el in elements:
        if el["type"] == "heading":
            level = el["level"]

            # New heading = new section boundary -> flush whatever text we've
            # accumulated under the PREVIOUS heading before moving on.
            if buffer_words:
                flush_buffer()

            # Update the heading stack: drop deeper/equal levels, push the new one
            heading_stack = [h for h in heading_stack if h["level"] < level]
            heading_stack.append({"level": level, "text": el["text"]})

        elif el["type"] == "paragraph":
            words = el["text"].split()
            buffer_words.extend(words)

            if len(buffer_words) >= MAX_CHUNK_WORDS:
                flush_buffer()

        elif el["type"] == "table":
            # Tables are never split or merged with surrounding paragraph text --
            # each table becomes its own chunk, tagged with the section it's under.
            if buffer_words:
                flush_buffer()
            table_text = table_to_text(el["rows"])
            if table_text:
                path = section_path_str(heading_stack)
                full_text = f"[{path}]\n(Table)\n{table_text}" if path else f"(Table)\n{table_text}"
                chunks.append({
                    "id": next_id(),
                    "source": source,
                    "section_path": path,
                    "chunk_type": "table",
                    "text": full_text
                })

    if buffer_words:
        flush_buffer()

    # Merge any very short trailing chunks into the previous chunk from the
    # same document, so we don't end up with tiny near-empty fragments.
    merged = []
    for chunk in chunks:
        if (merged and chunk["chunk_type"] == "text"
                and merged[-1]["chunk_type"] == "text"
                and len(chunk["text"].split()) < MIN_CHUNK_WORDS):
            merged[-1]["text"] += "\n" + chunk["text"]
        else:
            merged.append(chunk)

    return merged


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"Folder '{INPUT_FOLDER}' not found. Run Step 2 first.")
        return

    json_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".json")]

    if not json_files:
        print(f"No .json files found in '{INPUT_FOLDER}'. Run Step 2 first.")
        return

    all_chunks = []
    global_id = 0

    for filename in json_files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        doc_chunks = chunk_document(doc_data)

        # re-assign globally unique sequential ids while keeping the readable id too
        for c in doc_chunks:
            c["id"] = global_id
            global_id += 1

        all_chunks.extend(doc_chunks)
        print(f"{filename}: {len(doc_chunks)} chunks "
              f"({sum(1 for c in doc_chunks if c['chunk_type']=='table')} tables)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()