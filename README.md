# 🔗 GraphRAG — Knowledge Graph Enhanced Q&A

A high-performance **GraphRAG application** combining vector similarity search with knowledge graph reasoning, powered by **Neo4j AuraDB (Cloud)**, **embedded local Qdrant**, and a **modern HTML5/CSS3/Vanilla JS web dashboard** with interactive `vis-network` physics visualization.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Neo4j AuraDB](https://img.shields.io/badge/Neo4j-AuraDB_Cloud-008CC1)
![Qdrant](https://img.shields.io/badge/Qdrant-Embedded_Local-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML5_CSS3_Vanilla_JS-orange)

---

## 🚀 Architecture (Zero-Docker)

```
PDF / Text Upload ──► PyMuPDF Parsing ──► SentenceSplitter Chunking
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
 Embedded Qdrant (Vectors)        Neo4j AuraDB (Knowledge Graph)
   (data/qdrant_db)                  (neo4j+s:// cloud)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
         Hybrid Retrieval: Vector Top-K + Graph Traversal
                             │
                             ▼
               Gemini 2.5 Flash / GPT-4o
                             │
                             ▼
    Grounded Response + Citations + Interactive Graph View
```

---

## ⚡ Features

- **Zero-Docker Setup**: Vector search runs in-process via embedded Qdrant (`data/qdrant_db`) and Knowledge Graph connects to Neo4j AuraDB Cloud.
- **Async Extraction & Ingestion**: Non-blocking concurrent entity & relation extraction with automatic quota fallback.
- **Batched AuraDB Operations**: Fast `UNWIND` Cypher batching (100+ entities merged in <1.5s).
- **Interactive Force-Directed Graph**: Built with `vis-network` (canvas physics, zoom/pan, node click inspection, color-coded node categories, and document subgraph exploration).
- **Hybrid Retrieval Modes**: Easily toggle between **Graph-Enhanced** and **Vector-Only** modes.
- **Source Citations**: Collapsible source passages with exact filenames, page numbers, and relevance scores.
- **Cascade Deletion**: Cleanly delete documents with automatic removal from SQLite, Qdrant vectors, and Neo4j graph nodes.

---

## 🛠️ Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/yuvrajgovindrao/GraphRAG.git
cd GraphRAG

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell 
.venv\Scripts\activate

if you get a script execution policy error, run this once: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Linux / macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # Linux / macOS
```

Edit `.env`:
```env
# LLM Provider ("gemini" or "openai")
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j AuraDB (Cloud)
NEO4J_URI=neo4j+s://<your-instance-id>.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_auradb_password
```

### 3. Run the Application

Start the FastAPI server directly with Uvicorn:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

> [!TIP]
> You can also run directly without activating the virtual environment:
> ```bash
> .venv\Scripts\uvicorn.exe backend.main:app --reload --host 127.0.0.1 --port 8000
> ```

### 4. Open the Web Dashboard

Open your browser and navigate to:
- **Web Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

> [!IMPORTANT]
> Always visit **[http://localhost:8000](http://localhost:8000)** (or **http://127.0.0.1:8000**) in your web browser.
> *(Do not type `0.0.0.0` in browser address bars as browsers cannot resolve `0.0.0.0`).*

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service connectivity status (AuraDB, Qdrant, SQLite, LLM) |
| `POST` | `/upload` | Upload PDF or TXT file for background ingestion |
| `GET` | `/documents` | List all uploaded documents with status |
| `GET` | `/documents/{id}` | Get document metadata |
| `GET` | `/documents/{id}/status` | Poll ingestion & extraction progress |
| `GET` | `/documents/{id}/chunks` | Inspect document chunks and page mappings |
| `DELETE` | `/documents/{id}` | Cascade delete document, vectors, and graph entities |
| `POST` | `/query` | Ask questions in `graph` or `vector` mode |
| `GET` | `/graph/summary` | Knowledge graph entity and relationship counts |
| `GET` | `/graph/document/{id}` | Subgraph nodes and edges for visualizer |
| `POST` | `/evaluate` | Run evaluation suite comparing retrieval modes |

---

## 📂 Project Structure

```
graphrag-app/
├── backend/
│   ├── config.py             # Pydantic Settings & environment config
│   ├── database.py           # SQLite metadata layer
│   ├── models.py             # Pydantic request/response models
│   ├── llm_provider.py       # Asynchronous Gemini & OpenAI provider
│   ├── main.py               # FastAPI application & static file mount
│   ├── ingestion/
│   │   ├── parser.py         # PyMuPDF document parser
│   │   └── chunker.py        # SentenceSplitter chunker with page tracking
│   ├── extraction/
│   │   ├── prompts.py        # Entity & relationship extraction prompts
│   │   └── extractor.py      # Async batch extraction engine
│   ├── graph/
│   │   ├── neo4j_client.py   # Neo4j AuraDB client with UNWIND batching
│   │   ├── entity_resolution.py # Entity deduplication & normalization
│   │   └── traversal.py      # Bidirectional graph expansion
│   ├── retrieval/
│   │   ├── vector_search.py  # Local embedded Qdrant vector engine
│   │   └── graph_rag.py      # Hybrid graph-enhanced retrieval
│   └── evaluation/
│       └── evaluator.py      # Evaluation & benchmarking runner
├── frontend/
│   ├── index.html            # Single-page web application
│   ├── css/
│   │   └── style.css         # Modern dark glassmorphic styling
│   └── js/
│       └── app.js            # App logic & vis-network graph renderer
├── data/
│   └── raw/                  # Uploaded document storage
├── eval/
│   └── test_questions.json   # Evaluation test set
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```
