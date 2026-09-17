"""
Generates a synthetic evaluation testset using RAGAS, built directly from your
own chunks.json (the same chunks your RAG pipeline actually retrieves from).

RAGAS uses an LLM to read your documents and synthesize realistic questions,
along with ground-truth answers and the source context each question came from.

Install first:
    pip install ragas langchain-google-genai
"""

import os

# ragas imports GitPython internally for an experiment-tracking feature we
# don't use here. On machines without a git executable available/in PATH,
# that import fails loudly by default -- this silences it since it's not
# needed for testset generation.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import json
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms

load_dotenv()

CHUNKS_FILE = "chunks.json"
OUTPUT_FILE = "eval_dataset.csv"
KG_FILE = "knowledge_graph.json"   # persisted knowledge graph, reusable across runs
TESTSET_SIZE = 60

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError(
        "No API key found. Add this to your .env file:\n"
        "  GOOGLE_API_KEY=your_key_here"
    )


EXTRACTED_STRUCTURE_FOLDER = "extracted_structure"   # output of extract_text.py (Step 2)


def table_to_markdown(rows):
    if not rows:
        return ""
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def elements_to_markdown(elements):
    """Reconstruct a full document as real Markdown (# / ## headings, tables)
    from the structured elements saved by extract_text.py. RAGAS's own
    HeadlineExtractor/HeadlineSplitter transforms need actual heading syntax
    to detect document structure -- our pre-chunked, flattened chunks.json
    pieces don't have that, which is what caused the 'headlines property not
    found' error."""
    lines = []
    for el in elements:
        if el["type"] == "heading":
            level = max(1, el["level"])  # title (level 0) -> "#"
            lines.append(f"{'#' * level} {el['text']}")
        elif el["type"] == "paragraph":
            lines.append(el["text"])
        elif el["type"] == "table":
            lines.append(table_to_markdown(el["rows"]))
        lines.append("")  # blank line between blocks
    return "\n".join(lines)


def load_documents():
    """Load your ORIGINAL policy documents (one per source file, reconstructed
    as Markdown) for RAGAS to build its knowledge graph from -- not the
    pre-chunked chunks.json, which lacks the heading structure RAGAS needs
    to do its own internal splitting."""
    if not os.path.exists(EXTRACTED_STRUCTURE_FOLDER):
        raise FileNotFoundError(
            f"'{EXTRACTED_STRUCTURE_FOLDER}' not found. Run extract_text.py (Step 2) first."
        )

    docs = []
    for filename in os.listdir(EXTRACTED_STRUCTURE_FOLDER):
        if not filename.lower().endswith(".json"):
            continue
        with open(os.path.join(EXTRACTED_STRUCTURE_FOLDER, filename), "r", encoding="utf-8") as f:
            doc_data = json.load(f)

        markdown = elements_to_markdown(doc_data["elements"])
        docs.append(Document(
            page_content=markdown,
            metadata={"source": doc_data["source"], "title": doc_data.get("title", "")}
        ))

    return docs


def build_or_load_knowledge_graph(docs, generator_llm, generator_embeddings):
    """Builds the RAGAS knowledge graph (documents -> extracted entities,
    themes, and relationships between chunks), or loads a previously saved
    one so you don't have to re-run the (slow, LLM-driven) enrichment step
    every time you want to generate a fresh testset from the same docs."""

    if os.path.exists(KG_FILE):
        print(f"Loading existing knowledge graph from '{KG_FILE}'...")
        return KnowledgeGraph.load(KG_FILE)

    print("No existing knowledge graph found — building a new one "
          "(this reads through your docs with the LLM and can take a while)...")

    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
        )

    # Enriches the graph: extracts entities/themes/summaries per chunk and
    # builds relationships between related chunks (e.g. shared topics across
    # policies) -- this is what lets RAGAS generate multi-hop / cross-document
    # questions, not just single-chunk lookups.
    transforms = default_transforms(
        documents=docs, llm=generator_llm, embedding_model=generator_embeddings
    )
    apply_transforms(kg, transforms)

    kg.save(KG_FILE)
    print(f"Knowledge graph built and saved to '{KG_FILE}' "
          f"({len(kg.nodes)} nodes, {len(kg.relationships)} relationships).")

    return kg


def main():
    if not os.path.exists(EXTRACTED_STRUCTURE_FOLDER):
        print(f"'{EXTRACTED_STRUCTURE_FOLDER}' not found. Run extract_text.py first.")
        return

    docs = load_documents()
    print(f"Loaded {len(docs)} chunks as source documents for testset generation.")

    generator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", google_api_key=API_KEY)
    )
    generator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=API_KEY)
    )

    knowledge_graph = build_or_load_knowledge_graph(docs, generator_llm, generator_embeddings)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=knowledge_graph
    )

    print(f"Generating {TESTSET_SIZE} synthetic questions from the knowledge graph...")
    testset = generator.generate(testset_size=TESTSET_SIZE)

    df = testset.to_pandas()

    # RAGAS's default columns: user_input, reference, reference_contexts, synthesizer_name
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Saved {len(df)} questions to '{OUTPUT_FILE}'.")
    print("\nColumns generated:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df[["user_input", "reference"]].head(3).to_string())
    print(f"\nKnowledge graph saved separately at '{KG_FILE}' -- re-running this script "
          f"will reuse it instantly instead of rebuilding, unless you delete that file "
          f"(e.g. after adding new policy docs).")
    print("\nReview the CSV before using it as ground truth — synthetic questions"
          " occasionally need light editing for realism or accuracy.")


if __name__ == "__main__":
    main()