# 🤖 Hybrid RAG Chatbot

A fully **local, private** Retrieval-Augmented Generation (RAG) chatbot. Upload a PDF, index it, and ask questions about it in a chat interface. Answers are grounded in your document and cite the source page numbers.

Everything runs on your machine: embeddings and generation are served by [Ollama](https://ollama.com), so **no data leaves your computer and no API keys are required**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/Framework-LangChain-1C3C3C)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)

---

## ✨ Features

- 📄 **PDF upload and parsing** with page-level metadata preserved
- ✂️ **Chunking** of document text for precise retrieval
- 🧠 **Local embeddings** using `mxbai-embed-large` via Ollama
- 🗄️ **ChromaDB** vector store for semantic similarity search
- 💬 **Chat UI** built with Streamlit, with persistent conversation history per session
- 📌 **Grounded answers**: the LLM is instructed to answer only from retrieved context, say so when the answer isn't in the document, and cite page numbers
- 🔒 **100% local inference** with `llama3.2` running through Ollama

---

## 🏗️ How It Works

```
 ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
 │ Upload   │ -> │ Extract text  │ -> │ Chunk text   │ -> │ Embed chunks │
 │ PDF      │    │ (pypdf)       │    │ (+ metadata) │    │ (mxbai-embed)│
 └──────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘
                                                                 │
                                                                 ▼
 ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
 │ Answer + │ <- │ LLM generates │ <- │ Top-k chunks │ <- │ ChromaDB     │
 │ sources  │    │ (llama3.2)    │    │ as context   │    │ similarity   │
 └──────────┘    └───────────────┘    └──────────────┘    │ search       │
                                                          └──────────────┘
```

1. **Ingestion**: the uploaded PDF is parsed and split into chunks, each tagged with its source file and page number.
2. **Indexing**: each chunk is embedded with `mxbai-embed-large` and stored in ChromaDB.
3. **Retrieval**: at question time, the top **4** most semantically similar chunks are retrieved (`k=4`).
4. **Generation**: the chunks are passed to `llama3.2` (temperature `0` for deterministic, factual output) with a strict prompt to answer only from context and cite pages.

> **Note on "hybrid":** the project includes the dependencies for hybrid retrieval (`rank-bm25` for keyword search, `sentence-transformers` for reranking/embeddings, `langchain-classic` for ensemble retrievers). The current `app.py` uses dense vector similarity search; combining it with BM25 is the natural next step (see [Roadmap](#-roadmap)).

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://www.langchain.com) (`langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-classic`) |
| LLM | `llama3.2:latest` via [Ollama](https://ollama.com) (`langchain-ollama`) |
| Embeddings | `mxbai-embed-large:latest` via Ollama |
| Vector store | [ChromaDB](https://www.trychroma.com) (`langchain-chroma`) |
| PDF parsing | `pypdf` |
| Keyword search / reranking | `rank-bm25`, `sentence-transformers` |

---

## 📁 Project Structure

```
.
├── app.py                # Streamlit app: upload, index, and chat
├── src/
│   ├── ingestion.py      # load_and_chunk_pdf(): PDF -> chunks with metadata
│   └── vector_store.py   # create_vector_store(): embeds chunks into ChromaDB
├── test_app.py           # Minimal Streamlit smoke test (is the UI rendering?)
├── test_rag.py           # End-to-end pipeline test (Ollama + Chroma + LLM)
├── requirements.txt      # Python dependencies
└── README.md
```

---

## ✅ Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running
- Enough RAM to run `llama3.2` locally (8 GB+ recommended)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Ollama models

```bash
ollama pull llama3.2:latest
ollama pull mxbai-embed-large:latest
```

Make sure the Ollama service is running (it starts automatically after installation on most systems, or run `ollama serve`).

### 5. Run the app

```bash
python -m streamlit run app.py
```

Streamlit will open the app in your browser, usually at `http://localhost:8501`.

---

## 💡 Usage

1. **Upload** a PDF using the file uploader.
2. Click **Index PDF**. The app extracts text, creates embeddings, and builds the vector index.
3. Once you see *"Your chatbot is ready!"*, ask questions in the chat box.
4. Answers are based only on your document and reference the relevant page numbers.

Indexing a new PDF resets the chat history.

---

## 🧪 Testing Your Setup

Two small scripts help you diagnose problems quickly:

| Script | What it verifies | Command |
|---|---|---|
| `test_app.py` | Streamlit is installed and renders correctly | `python -m streamlit run test_app.py` |
| `test_rag.py` | Ollama embeddings, ChromaDB indexing, retrieval, and LLM generation all work end to end | `python test_rag.py` |

A successful `test_rag.py` run ends with:

```
ANSWER: The capital of France is Paris.
SUCCESS: End-to-end RAG pipeline works!
```

---

## ⚙️ Configuration

Key parameters live directly in the code and are easy to tune:

| Parameter | Location | Default | Effect |
|---|---|---|---|
| LLM model | `app.py` (`ChatOllama`) | `llama3.2:latest` | Swap for any model you have pulled in Ollama |
| Temperature | `app.py` | `0` | Higher = more creative, lower = more factual |
| Retrieved chunks (`k`) | `app.py` (`similarity_search`) | `4` | More context vs. more noise |
| Embedding model | `src/vector_store.py` | `mxbai-embed-large:latest` | Must be pulled in Ollama |

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` / Ollama errors | Make sure Ollama is running (`ollama serve`) |
| `model not found` | Run `ollama pull llama3.2:latest` and `ollama pull mxbai-embed-large:latest` |
| PowerShell blocks venv activation | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then retry |
| Empty or poor answers on a scanned PDF | The PDF likely has no text layer; run OCR on it first |
| Slow responses | Use a smaller model, reduce `k`, or close other memory-heavy apps |

---

## 🗺️ Roadmap

- [ ] **True hybrid retrieval**: combine BM25 keyword search with dense vector search (ensemble retriever)
- [ ] **Reranking** of retrieved chunks with a cross-encoder (`sentence-transformers`)
- [ ] Streaming token-by-token responses
- [ ] Multi-PDF support and persistent vector store across sessions
- [ ] Show retrieved source snippets alongside each answer
- [ ] Retrieval evaluation (hit rate / MRR) and answer-faithfulness metrics
- [ ] Conversation-aware question rewriting
- [ ] Dockerfile for one-command setup

---

## ⚠️ Limitations

- Answer quality depends on the local model size; small models can miss nuance in long or technical documents.
- Currently supports one indexed PDF at a time.
- Scanned/image-only PDFs need OCR before indexing.
- The conversation history is displayed but is not fed back into the model, so each question is answered independently.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 👤 Author

**Your Name**
GitHub: [@your-username](Abhisek Samal)

---

⭐ If you found this project useful, consider giving it a star!
