# 🔍 Multi-Document Research Assistant

A fully local, privacy-preserving RAG (Retrieval-Augmented Generation) application that lets you upload PDFs, Word documents, text files, and Markdown, then ask natural-language questions and receive answers grounded in your own documents — with source citations.

---

## ✨ Features

| Capability | Detail |
|---|---|
| **Document formats** | PDF, DOCX, TXT, Markdown (`.md`) |
| **Embeddings** | `sentence-transformers` — runs fully locally, no API key needed |
| **Vector store** | ChromaDB (persistent on disk, content-hash deduplication) |
| **LLM** | OpenAI GPT-4.1 by default; swap any Chat-Completions model in `.env` |
| **Chunking** | Recursive character splitter with configurable size & overlap |
| **Retrieval** | Semantic k-NN search; optional score threshold |
| **Memory** | Sliding-window conversation history (configurable turns) |
| **UI** | Streamlit chat interface with source-citation pills |
| **Logging** | Colourised console + rotating file logger |

---

## 🗂 Project Structure

```
rag_project/
│
├── app.py                  # Streamlit entry point
├── requirements.txt
├── README.md
├── .env.example            # Copy → .env and add your OpenAI key
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Pydantic-Settings configuration
│   ├── logger.py           # Rotating file + coloured console logging
│   ├── loaders.py          # PDF / DOCX / TXT / MD loaders
│   ├── chunker.py          # RecursiveCharacterTextSplitter wrapper
│   ├── embeddings.py       # SentenceTransformer embedding model
│   ├── vector_store.py     # ChromaDB persistence layer
│   ├── retriever.py        # Semantic retriever (BaseRetriever)
│   ├── llm.py              # ChatOpenAI wrapper
│   ├── rag_chain.py        # Orchestration: retrieve → prompt → generate
│   └── utils.py            # File I/O, source formatting, helpers
│
├── data/                   # Uploaded files (auto-created, git-ignored)
├── chroma_db/              # ChromaDB persistence (auto-created, git-ignored)
└── logs/                   # Rotating log files (auto-created, git-ignored)
```

---

## 🚀 Quick Start

### 1 — Clone / copy the project

```bash
git clone <your-repo-url> rag_project
cd rag_project
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users** — replace `torch>=2.2.0` in `requirements.txt` with the
> appropriate CUDA-enabled wheel from https://pytorch.org/get-started/locally/

### 4 — Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your OpenAI API key:

```dotenv
OPENAI_API_KEY=sk-...
```

All other values have sensible defaults and can be left unchanged for a first run.

### 5 — Run the application

```bash
streamlit run app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically.

---

## 🖥 Usage

1. **Upload documents** — drag and drop one or more PDF / DOCX / TXT / MD files into the sidebar.  
2. **Index** — click **⚡ Index Documents**.  The progress bar shows chunking and embedding in real time.  
3. **Ask questions** — type any question in the chat input at the bottom.  
4. **View sources** — every answer shows which files (and chunks) the LLM used.  
5. **New Chat** — clear conversation history without losing the knowledge base.  
6. **Clear KB** — wipe the vector store and all uploaded files to start fresh.

---

## ⚙️ Configuration Reference

All settings live in `.env`.  The full list with defaults:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4.1` | Chat-Completions model name |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `1024` | Max response tokens |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION_NAME` | `research_assistant` | Collection name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Documents retrieved per query |
| `MAX_HISTORY_TURNS` | `6` | Conversation turns in context |
| `DATA_PATH` | `./data` | Upload staging directory |
| `LOG_PATH` | `./logs` | Log file directory |
| `LOG_LEVEL` | `INFO` | Python log level |

### Swapping the LLM

Change one line in `.env`:

```dotenv
LLM_MODEL=gpt-4o-mini   # cheaper & faster
# LLM_MODEL=gpt-3.5-turbo
```

No code changes required.

---

## 🔧 Architecture

```
User question
      │
      ▼
┌─────────────────┐    top-k chunks    ┌──────────────┐
│   RAGChain      │◄──────────────────►│  Retriever   │
│  (orchestrator) │                    │  (semantic   │
│                 │                    │   search)    │
└────────┬────────┘                    └──────┬───────┘
         │                                    │
         │  context block                     │
         │  + history                         ▼
         ▼                             ┌──────────────┐
    ┌─────────┐                        │  VectorStore │
    │   LLM   │                        │  (ChromaDB)  │
    │ (OpenAI)│                        └──────┬───────┘
    └────┬────┘                               │
         │                                    │  indexed chunks
         │  answer                            ▼
         ▼                             ┌──────────────┐
    Streamlit UI  ◄──sources───────────│   Chunker    │
                                       └──────┬───────┘
                                              │  split docs
                                              ▼
                                       ┌──────────────┐
                                       │   Loaders    │
                                       │  PDF/DOCX/   │
                                       │  TXT/MD      │
                                       └──────────────┘
```

---

## 🧪 Running a Quick Sanity Check

```python
# From the project root with .venv active
python - <<'EOF'
from src.config import settings
from src.embeddings import EmbeddingModel

print("Settings OK — model:", settings.LLM_MODEL)
em = EmbeddingModel(settings.EMBEDDING_MODEL)
vec = em.embed_query("Hello world")
print(f"Embedding OK — {len(vec)}-dim vector")
EOF
```

---

## 📝 Logging

Logs are written to `./logs/rag_assistant.log` (rotating, 5 × 5 MB).  
Console output is colourised by severity.  
Adjust verbosity via `LOG_LEVEL=DEBUG` in `.env`.

---

## 📄 License

MIT — use freely, attribute appreciated.
