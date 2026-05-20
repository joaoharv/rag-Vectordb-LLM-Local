# RAG API with Ollama & ChromaDB

A fully local **Retrieval-Augmented Generation (RAG)** API built with FastAPI, ChromaDB, and Ollama. No cloud dependencies — all inference runs on your own machine.

---

## Overview

This project implements the RAG pattern from scratch:

1. **Index** — text documents are split into chunks and stored as vector embeddings in ChromaDB
2. **Retrieve** — when a question arrives, the most semantically similar chunks are fetched
3. **Generate** — the retrieved context is injected into a prompt and sent to a local LLM via Ollama

```
User question
     │
     ▼
┌──────────────┐     vector search     ┌─────────────────┐
│  FastAPI     │ ─────────────────────▶│   ChromaDB      │
│  /ask        │ ◀─────────────────────│  (chroma_db/)   │
└──────────────┘   top-k chunks        └─────────────────┘
     │
     │ augmented prompt
     ▼
┌──────────────┐
│   Ollama     │  llama3.1  (LLM)
│              │  nomic-embed-text  (embeddings)
└──────────────┘
     │
     ▼
  JSON response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) |
| LLM & Embeddings | [Ollama](https://ollama.com/) |
| LLM Model | `llama3.1` |
| Embedding Model | `nomic-embed-text` |
| Validation | Pydantic v2 |

---

## Features

- **Fully local** — no OpenAI, no API keys, no data leaving your machine
- **Persistent knowledge base** — ChromaDB stores vectors on disk across restarts
- **Dynamic document ingestion** — add or update profiles via REST API
- **Idempotent upserts** — resubmitting the same user safely replaces previous chunks
- **Source transparency** — every answer includes the context chunks used to generate it
- **Input validation** — empty or whitespace-only requests are rejected with proper HTTP 400 errors

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed and running

Pull the required models before starting:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/RAG-Org-Ollama.git
cd RAG-Org-Ollama

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Build the initial knowledge base

Place your content in `profile.txt` (paragraphs separated by blank lines), then run:

```bash
python build_knowledge_base.py
```

Expected output:
```
Loaded 4 chunks from profile.txt
Added 4 chunks to the 'personal_profile' collection.
Knowledge base built successfully!
```

### 2. Start the API server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

### `GET /ask`

Query the knowledge base with a natural language question.

**Query Parameters**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | The question to answer |

**Example**

```bash
curl "http://localhost:8000/ask?question=What%20are%20his%20career%20goals?"
```

**Response**

```json
{
  "question": "What are his career goals?",
  "answer": "His career goal is to become an AI engineer, with a focus on AI automations and RAG systems.",
  "context_used": [
    "My career goal is to become a AI engineer.\nI'm especially interested in AI automations and RAG!.",
    "I'm currently learning about cloud computing, AI, and DevOps."
  ]
}
```

---

### `POST /documents`

Add or update a user profile in the knowledge base. If the user already exists, their previous chunks are replaced.

**Request Body**

```json
{
  "user_name": "joao",
  "content": "Paragraph one about the user.\n\nParagraph two about the user."
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `user_name` | string | Yes | Unique identifier for the document owner |
| `content` | string | Yes | Full profile text (paragraphs separated by blank lines) |

**Example**

```bash
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"user_name": "joao", "content": "I am a software engineer.\n\nI specialize in AI and backend development."}'
```

**Response**

```json
{
  "message": "Added 2 chunks for user 'joao'",
  "user_name": "joao",
  "chunks_added": 2
}
```

---

## Project Structure

```
RAG-Org-Ollama/
│
├── main.py                 # FastAPI application (REST API + RAG pipeline)
├── build_knowledge_base.py # One-time script to seed ChromaDB from profile.txt
├── profile.txt             # Sample knowledge base document
├── requirements.txt        # Python dependencies
├── chroma_db/              # Persisted vector database (auto-generated)
└── README.md
```

---

## How RAG Works

Traditional LLMs answer questions based solely on their training data. RAG extends this by:

1. **Storing your own documents** as vector embeddings (numerical representations of meaning)
2. **Searching by semantic similarity** — not keyword matching — to find the most relevant passages
3. **Grounding the LLM response** in that retrieved context, reducing hallucinations and enabling answers about private data

This project uses `nomic-embed-text` to generate embeddings and `llama3.1` as the reasoning model, both running locally through Ollama.

---

## License

MIT
