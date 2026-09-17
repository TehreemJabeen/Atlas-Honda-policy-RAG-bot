# Company Policy RAG Chatbot

An internal chatbot that answers employee questions about company policies (leave, benefits,
conduct, remote work, and more) using Retrieval-Augmented Generation (RAG). Instead of relying
on a language model's general knowledge, the system retrieves the actual relevant policy content
and grounds every answer in it — preventing hallucinated policy details and keeping every answer
traceable back to a source document and section.

## Features

- **Structure-aware document ingestion** — preserves headings, sections, and tables from `.docx`
  policy files instead of flattening them to plain text
- **Hierarchical chunking** — splits along section boundaries, keeps tables intact, tags every
  chunk with its section breadcrumb
- **Incremental embeddings** — hash-based change detection re-embeds only new/modified chunks
- **Relevance-ranked retrieval with source diversity cap** — avoids both near-duplicate results
  and the diversity-penalty pitfalls of naive MMR search
- **Query rewriting + multi-query retrieval (RRF)** — normalizes casual phrasing and searches
  multiple related phrasings, fused via Reciprocal Rank Fusion, for consistent results regardless
  of how a question is worded
- **Structured, schema-validated answers** — responses are broken into distinct scenarios with
  eligibility rules, action steps, exceptions, and citations, not a single wall of text
- **Streamlit UI** — branded chat interface with expandable scenario cards and a source-transparency
  debug panel

## Architecture

```
Policy docs (.docx)
        │
        ▼
Extract structure (headings, sections, tables)  ──  extract_text.py
        │
        ▼
Hierarchical chunking (section-aware, table-preserving)  ──  chunk_text.py
        │
        ▼
Embed + store (incremental updates)  ──  generate_embeddings.py
        │
        ▼
Retrieve (relevance + source cap, multi-query RRF)  ──┐
        │                                              │  chatbot.py
Generate structured answer (LangChain + Gemini)  ──────┘
        │
        ▼
Streamlit UI  ──  app.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM & embeddings | Google Gemini API |
| Orchestration | LangChain |
| Vector store | LangChain `InMemoryVectorStore` (pure Python, no native dependencies) |
| Document extraction | `python-docx` |
| Structured output | Pydantic |
| UI | Streamlit |

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your Gemini API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
3. Place your policy `.docx` files in a `Policies/` folder.

## Usage

Run the pipeline in order:

```
python extract_text.py         # Step 1: extract document structure
python chunk_text.py           # Step 2: hierarchical chunking
python generate_embeddings.py  # Step 3: generate/update embeddings
python chatbot.py              # Step 4: CLI chat (optional, for testing)
streamlit run app.py           # Step 5: web UI
```

## Project Structure

```
├── extract_text.py          # Structure-aware .docx extraction
├── chunk_text.py             # Hierarchical, table-aware chunking
├── generate_embeddings.py    # Embedding generation with incremental updates
├── chatbot.py                 # Retrieval + structured generation (core RAG logic)
├── app.py                     # Streamlit UI
├── requirements.txt
├── .gitignore
└── README.md
```

## Roadmap

- [ ] Systematic evaluation pipeline (RAGAS-based synthetic testset)
- [ ] Hybrid search (vector + BM25 keyword search)
- [ ] Cross-encoder reranking
- [ ] Automated email response agent (LangGraph-based, confidence-routed)

## Notes

Generated pipeline artifacts (`extracted_structure/`, `extracted_text/`, `vector_store.json`,
`chroma_db/`) are excluded from version control via `.gitignore` since they're regenerable from
the source policy documents and scripts.
