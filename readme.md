# PDF Question-Answering with FAISS & GPT-4

This project allows you to index PDF documents using FAISS and query them with GPT-4 to retrieve relevant answers. It leverages **LangChain**, **HuggingFace Embeddings**, and **Sentence Transformers**.

---

## Features

- Load and split PDF documents automatically.
- Create a FAISS vector index from document embeddings.
- Query indexed documents for relevant chunks.
- Generate context-aware responses with GPT-4.

---

## Requirements

- Python 3.10+
- [FAISS](https://github.com/facebookresearch/faiss)
- `langchain`, `langchain_openai`, `langchain_huggingface`, `langchain_community`
- `sentence-transformers`
- `torch`
- `PyPDF2` or other PDF loaders if needed

Install dependencies:

```bash
pip install faiss-cpu langchain langchain_openai langchain_huggingface langchain_community sentence-transformers torch
````

---

## Project Structure

```
.
├── docs/                  # Folder containing PDF documents to index
├── faiss_index/           # Directory where FAISS index will be saved
├── main.py                # Main script containing indexing and querying logic
```

---

## Usage

### 1. Indexing Documents

Before querying, you need to create the FAISS index.

```python
from your_module import indexMaker, HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
indexMaker(embeddings)  # Creates and saves 'faiss_index' locally
```

> ⚠️ Currently, the script **does not check if the FAISS index already exists**. Run `indexMaker()` only when you want to rebuild the index.

---

### 2. Querying the Index

Once the FAISS index is created, you can query it for relevant document chunks:

```python
from your_module import indexLoader, gpt4, HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
question = "What are the directions I can move in with the knight piece?"

vecResult = indexLoader(embeddings, question)  # Returns top relevant chunks
response = gpt4(vecResult, question)          # Generates GPT-4 response
print(response.content)
```

---

## How It Works

1. **Load PDFs**: Uses `DirectoryLoader` to read all PDFs in `docs/`.
2. **Split Documents**: Uses `RecursiveCharacterTextSplitter` to create manageable chunks.
3. **Embed & Index**: Converts text chunks into embeddings and stores them in FAISS.
4. **Query**: Searches the FAISS index for top-k similar chunks.
5. **Answer**: Passes retrieved chunks to GPT-4 to generate a context-aware answer.

---

## Notes

* Replace `openai_api_key` in `gpt4()` with your actual API key.
* FAISS index must exist before querying; run `indexMaker()` first.
* You can adjust `chunk_size` and `chunk_overlap` in the splitter for finer control of embeddings.

---
