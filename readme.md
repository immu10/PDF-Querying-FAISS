# PDF querying Faiss

A lightweight Retrieval-Augmented Generation (RAG) pipeline that turns a folder of PDFs into a searchable knowledge base. Documents are embedded with HuggingFace sentence-transformers, indexed in a FAISS vector store, and queried through GPT-4o to return grounded, context-aware answers.

---

## Overview

This project ingests PDF files from a local `docs/` directory, splits them into overlapping text chunks, and stores their vector embeddings in a FAISS index saved to disk. At query time, the user's question is embedded and matched against the index to retrieve the most relevant chunks. Those chunks are passed as context to an OpenAI GPT-4o model, which generates an answer constrained to the retrieved material.

Key capabilities:

- **Automated PDF ingestion** — recursively loads every `*.pdf` under `docs/`.
- **Local vector store** — FAISS index is built once and persisted (`faiss_index/`) for fast reuse.
- **Semantic retrieval** — top-k similarity search using `all-mpnet-base-v2` embeddings.
- **Grounded generation** — GPT-4o answers strictly from the retrieved chunks.

---

## Architecture

```
            +-------------------+
            |   PDFs in docs/   |
            +-------------------+
                     |
                     v
        +-----------------------------+
        |  DirectoryLoader (PyPDF)    |   load PDF pages
        +-----------------------------+
                     |
                     v
        +-----------------------------+
        | RecursiveCharacterText      |   chunk_size=1200
        | Splitter -> text chunks     |   overlap=200
        +-----------------------------+
                     |
                     v
        +-----------------------------+
        | HuggingFace Embeddings      |   all-mpnet-base-v2
        | (sentence-transformers)     |
        +-----------------------------+
                     |
                     v
        +-----------------------------+
        |   FAISS IndexFlatL2         |   save_local("faiss_index")
        |   vector store on disk      |
        +-----------------------------+
                     |
        =============|=============  (build phase above / query phase below)
                     |
   user question     |
        |            v
        |   +-----------------------------+
        +-->|  similarity_search(k=2)     |   retrieve top chunks
            +-----------------------------+
                     |
                     v
        +-----------------------------+
        |  Prompt = question + chunks |
        +-----------------------------+
                     |
                     v
        +-----------------------------+
        |   ChatOpenAI (GPT-4o)       |   temperature=0
        +-----------------------------+
                     |
                     v
            +-------------------+
            |  Grounded answer  |
            +-------------------+
```

---

## Tech Stack

| Layer            | Technology                                | Purpose                                       |
| ---------------- | ----------------------------------------- | --------------------------------------------- |
| Language         | Python 3.10+                              | Core runtime                                  |
| Orchestration    | LangChain                                 | Loaders, splitters, vector-store integration  |
| Document loading | PyPDFLoader / DirectoryLoader             | Read and parse PDF files                       |
| Text splitting   | RecursiveCharacterTextSplitter            | Chunk documents (1200 chars, 200 overlap)     |
| Embeddings       | sentence-transformers (all-mpnet-base-v2) | Convert text chunks to vectors                |
| Vector store     | FAISS (faiss-cpu, IndexFlatL2)            | Index and similarity search                   |
| LLM              | OpenAI GPT-4o (langchain-openai)          | Generate grounded answers                     |
| Compute          | PyTorch                                   | Backend for the embedding model               |
| API (optional)   | FastAPI + Uvicorn                         | Serve the pipeline over HTTP                  |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-key-here"     # PowerShell: $env:OPENAI_API_KEY="your-key-here"
```

### 3. Add your PDFs

Drop any PDF files into the `docs/` directory.

### 4. Build the index (run once)

In [RAG.py](RAG.py), uncomment the `indexMaker(embeddings)` call, then run:

```bash
python RAG.py
```

This creates and saves the FAISS index to `faiss_index/`.

### 5. Query

Set your `question` in [RAG.py](RAG.py) and run the script again (with `indexMaker` re-commented) to retrieve chunks and print the GPT-4o answer.

---

## Project Structure

```
.
├── docs/              # PDF documents to index
├── faiss_index/       # Persisted FAISS vector store (generated)
├── RAG.py             # Core pipeline: load, split, embed, index, query, answer
├── test.py            # Quick test for the loader + splitter
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Notes

- The script does **not** check whether the FAISS index already exists — only run `indexMaker()` when you intend to (re)build it.
- Adjust `chunk_size` / `chunk_overlap` in `splitter()` and `k` in `indexLoader()` to tune retrieval quality.

---

## Tags

`RAG` · `LangGraph` · `FAISS` 
