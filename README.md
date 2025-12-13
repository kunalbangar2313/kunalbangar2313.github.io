# kunalbangar2313.github.io

## RAG App with FAISS and OpenAI

A Streamlit-based Retrieval-Augmented Generation (RAG) application that allows you to upload documents and ask questions about them.

### Features
- Upload .txt, .md, or .pdf files
- Persistent FAISS vectorstore caching (avoids repeated embedding calls)
- Optional local embeddings fallback to avoid OpenAI quota issues
- Question answering based on uploaded documents

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Basic Usage (OpenAI Embeddings)
```bash
export OPENAI_API_KEY="sk-..."
streamlit run rag_app.py
```

#### Using Local Embeddings (No OpenAI API Key Required)
To avoid OpenAI quota exhaustion or run completely offline:
```bash
export USE_LOCAL_EMBEDDINGS=1
streamlit run rag_app.py
```

This uses HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` model for embeddings instead of OpenAI.

### Caching
The app automatically caches vectorstores in `./.vectorstore/` directory based on document content fingerprints. If you upload the same documents again, the cached vectorstore will be loaded instantly without re-embedding.

To clear the cache:
```bash
rm -rf .vectorstore
```