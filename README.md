# 🔗 GraphRAG — Knowledge Graph Enhanced Q&A

A high-performance **GraphRAG application** combining vector similarity search with knowledge graph reasoning, powered by **Neo4j AuraDB (Cloud)**, **embedded local Qdrant**, and a **modern HTML5/CSS3/Vanilla JS web dashboard** with interactive `vis-network` physics visualization.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Neo4j AuraDB](https://img.shields.io/badge/Neo4j-AuraDB_Cloud-008CC1)
![Qdrant](https://img.shields.io/badge/Qdrant-Embedded_Local-red)
![Frontend](https://img.shields.io/badge/Frontend-HTML5_CSS3_Vanilla_JS-orange)

---

## 🏛️ System Architecture

GraphRAG utilizes a zero-Docker, modular hybrid architecture that couples dense vector semantic search with structured graph traversal and reasoning.

```mermaid
flowchart TB
    subgraph Client["🖥️ Presentation Layer (Modern Vanilla Web Dashboard)"]
        UI["Web Dashboard\n(HTML5 / CSS3 / ES6+ JS)"]
        VIS["Interactive Graph Arena\n(vis-network Physics Engine)"]
        UPLOAD["Multi-File Dropzone\n(Batch .pdf, .txt, .md)"]
        PROGRESS["Live Progress Tracker\n(Real-Time Chunks & Status)"]
        UI <--> VIS
        UI <--> UPLOAD
        UI <--> PROGRESS
    end

    subgraph API["⚡ Application & Ingestion Layer (FastAPI)"]
        ROUTER["REST API Controller\n(/upload, /query, /graph/*, /documents)"]
        SEMAPHORE["Ingestion Semaphore Queue\n(asyncio.Semaphore Concurrency Control)"]
        PARSER["Fault-Tolerant Document Parser\n(PyMuPDF with Per-Page Error Isolation)"]
        CHUNKER["Semantic Sentence Chunker\n(SentenceSplitter with Page Tracking)"]
        EXTRACTOR["Entity & Relation Extractor\n(Structured JSON Output Engine)"]
        RESOLVER["Entity Resolution & Canonicalization\n(Name Normalization & Reference Merge)"]
        RETRIEVER["Hybrid Graph-RAG Engine\n(Balanced Multi-Document Expansion)"]
    end

    subgraph Storage["💾 Multi-Model Storage Layer"]
        SQLITE[("SQLite\ndata/graphrag.db\n(Metadata, Chunks, Progress)")]
        QDRANT[("Embedded Qdrant\ndata/qdrant_db\n(Dense Chunk Vectors - 768d)")]
        NEO4J[("Neo4j AuraDB Cloud\nneo4j+s://\n(Entities & Typed Relationships)")]
    end

    subgraph AI["🧠 AI & LLM Services"]
        LLM["Google Gemini / OpenAI\n(gemini-2.5-flash / gpt-4o)"]
        EMBED["Embedding Engine\n(gemini-embedding-001 / text-embedding-3)"]
    end

    %% Ingestion Pipeline Flow
    UPLOAD ==>|Multipart Batch Upload| ROUTER
    ROUTER --> SEMAPHORE
    SEMAPHORE --> PARSER --> CHUNKER
    CHUNKER -->|Store Chunks & Stats| SQLITE
    CHUNKER -->|Generate Embeddings| EMBED
    EMBED -->|Upsert Dense Vectors| QDRANT
    CHUNKER -->|Text Chunks| EXTRACTOR
    EXTRACTOR <==>|Structured JSON| LLM
    EXTRACTOR --> RESOLVER
    RESOLVER -->|Batched UNWIND Cypher| NEO4J

    %% Query & Retrieval Flow
    UI <==>|User Query & Retrieval Mode| ROUTER
    ROUTER <==> RETRIEVER
    RETRIEVER <==>|Top-K Dense Passages| QDRANT
    RETRIEVER -->|Extract Chunk Entities| NEO4J
    RETRIEVER <==>|Balanced Per-Doc Graph Facts| NEO4J
    RETRIEVER <==>|Synthesized Context + Prompt| LLM
    RETRIEVER -->|Answer + Citations + Subgraph| ROUTER
```

---

## 🔄 End-to-End Workflow Flowchart

The system runs on two unified, resilient pipelines: **Knowledge Graph Ingestion** and **Balanced Hybrid Retrieval**.

### 1. Document Ingestion Pipeline Flowchart

```mermaid
flowchart TD
    Start([User Selects / Drops Multiple Files]) --> Upload[POST /upload Batch Endpoint]
    Upload --> RegDB[Save to Disk & Insert Records in SQLite as 'processing']
    RegDB --> UIRefresh[UI Immediately Displays Document Cards & Animated Progress Tracks]
    
    subgraph Queue["Background Pipeline Worker (asyncio.Semaphore = 1)"]
        RegDB -.-> Worker[Worker Picks Document From Queue]
        Worker --> ParseDoc[Parse Document with PyMuPDF]
        ParseDoc -->|Corrupted Page Check| PageCheck{Page Corrupted?}
        PageCheck -->|Yes| SkipPage[Log Warning & Continue Other Pages]
        PageCheck -->|No| ExtractPageText[Extract Clean Page Text]
        SkipPage --> SemanticChunk
        ExtractPageText --> SemanticChunk[Split Into Sentence-Bounded Chunks]
        
        SemanticChunk --> InsertSQLite[Store Chunk Records in SQLite]
        SemanticChunk --> EmbedChunks[Generate Embeddings via LLM Provider]
        EmbedChunks --> IndexQdrant[Index Vectors in Local Embedded Qdrant]
        
        IndexQdrant --> ExtractEntities[Batch Extract Entities & Relationships via LLM]
        ExtractEntities --> ProgressUpdate[Emit Live Progress: X/Y Chunks Processed]
        ProgressUpdate --> ResolveEntities[Resolve & Canonicalize Entities]
        ResolveEntities --> BatchCypher[Batched UNWIND Cypher Merge into Neo4j AuraDB]
        BatchCypher --> MarkReady[Update Document Status to 'ready' in SQLite]
    end
    
    MarkReady --> FinalUI[UI Updates to '✅ Ready' with Final Chunk Count]
```

### 2. Hybrid Graph-RAG Retrieval Flowchart (Balanced Multi-Document)

```mermaid
flowchart TD
    QueryInput([User Enters Question in Web Chat]) --> QueryReq[POST /query with mode='graph' & top_k]
    
    subgraph ParallelSearch["Phase 1: Multi-Document Vector Search & Seed Discovery"]
        QueryReq --> VectorSearch[Dense Vector Similarity Search in Qdrant]
        VectorSearch --> TopKChunks[Retrieve Top-K Chunks with doc_ids & filenames]
        
        QueryReq --> ExtractSeeds[Extract Key Entities from User Question via LLM]
        ExtractSeeds --> MatchGraphSeeds[Match Entities in Neo4j AuraDB]
        
        TopKChunks --> ChunkSeeds[Query Neo4j for Entities Linked to Retrieved chunk_ids]
        MatchGraphSeeds --> MergeSeeds[Merge Question Seeds & Chunk Entities]
        ChunkSeeds --> MergeSeeds
    end

    subgraph BalancedExpansion["Phase 2: Balanced Per-Document Graph Expansion"]
        MergeSeeds --> GroupDoc[Partition Seeds by Contributing Document]
        GroupDoc --> PerDocCheck{Multi-Document Query?}
        
        PerDocCheck -->|Yes| SubqueryExpansion["Execute Cypher with CALL (start) Subqueries\n(Allocate Even Quotas Across All Retrieved Documents)"]
        PerDocCheck -->|No| SingleDocExpansion["Standard Bidirectional 2-Hop Traversal\n(Expand up to max_facts = 45)"]
        
        SubqueryExpansion --> CollectFacts[Collect Balanced Graph Facts: Source —[REL]→ Target]
        SingleDocExpansion --> CollectFacts
    end

    subgraph AnswerGen["Phase 3: Hybrid Synthesis & Visualization"]
        TopKChunks --> AssemblePrompt[Assemble Comprehensive Hybrid Prompt]
        CollectFacts --> AssemblePrompt
        AssemblePrompt --> LLMGen[Generate Grounded Answer via Gemini / OpenAI]
        LLMGen --> ReturnPayload[Package Answer + Source Citations + Graph Subgraph]
    end

    ReturnPayload --> RenderChat[Render Markdown Answer with Collapsible Citations]
    ReturnPayload --> RenderGraph[Render Interactive vis-network Force-Directed Graph]
```

---

## ✨ Recent Updates & Changelog

- **🚀 Multi-File Batch Upload**: Select or drag-and-drop multiple `.pdf`, `.txt`, and `.md` files simultaneously. All files are enqueued and displayed immediately with individual status cards.
- **📊 Real-Time Chunk Counter & Live Progress Bars**:
  - Document Library header badge shows total document and chunk counts across the library (`X docs · Y chunks`).
  - Active progress tracks display real-time status: `⏳ Parsing` (indeterminate animation) → `🔍 Extracting KG` (live `X/Y chunks (Z%)` counter) → `✅ Ready` (final chunk count badge).
- **🛡️ Fault-Tolerant Pipeline Concurrency (`asyncio.Semaphore`)**:
  - Background ingestion worker processes documents sequentially, eliminating Gemini API `429 RESOURCE_EXHAUSTED` rate limits and SQLite database write locks.
  - Per-page parser resilience catches corrupted PDF pages individually without failing the overall document.
- **⚖️ Balanced Multi-Document Graph Expansion**:
  - Solved knowledge graph starvation on cross-document compound questions.
  - Integrated `find_entities_by_chunk_ids()` and Neo4j 5 `CALL (start) { ... }` subqueries to guarantee that every document retrieved by vector search receives an equal, rich share of graph facts and nodes on the canvas.
- **🌐 Full Cross-Document Knowledge Graph (`/graph/global`)**:
  - Added dedicated endpoint and dropdown selector to explore the entire connected knowledge graph across all uploaded documents.
  - Guaranteed endpoint integrity ensures 100% of relationship source/target nodes are rendered with zero broken links.
- **🎨 Visual Graph & Layout Polish**:
  - **Expandable Square View**: "Expand Graph" button switches to a full-width `95vh` square layout for wide-canvas graph exploration.
  - **Stable Centering**: View initializes centered upon render; removed delayed snap-back auto-fits.
  - **Standardized Header Heights**: Unified all panel headers to `62px`.
  - **Safe Error Handling**: Replaced strict JSON parsing with adaptive error decoding, preventing `Unexpected token 'I'` exceptions and displaying human-readable root causes directly in the UI.

---

## ⚡ Features

- **Zero-Docker Architecture**: Vector search runs in-process via embedded Qdrant (`data/qdrant_db`) and Knowledge Graph connects to Neo4j AuraDB Cloud.
- **Multi-File Batch Upload & Live Progress**: Select or drag-and-drop multiple documents simultaneously with live extraction progress bars, chunk counts, and status tracking.
- **Fault-Tolerant Concurrency Control**: Sequential pipeline semaphore prevents Gemini 429 quota exhaustion and SQLite lock contention; per-page parser resilience ensures damaged pages don't crash whole documents.
- **Batched AuraDB Operations**: Ultra-fast `UNWIND` Cypher batching (100+ entities merged in <1.5s) with guaranteed endpoint consistency and zero broken edges.
- **Global & Document Subgraphs**: View individual document graphs or inspect the entire connected Knowledge Graph across all uploaded documents.
- **Interactive Force-Directed Physics Arena**: Built with `vis-network` (tactile Barnes-Hut elasticity, straight arrow edges, expand-to-square full view, node inspector, and auto-centering).
- **Hybrid Retrieval Modes**: Easily toggle between **Graph-Enhanced** and **Vector-Only** Q&A modes.
- **Transparent Source Citations**: Collapsible source passages with exact filenames, page numbers, and relevance scores.
- **Cascade Deletion**: Cleanly delete documents with automatic cascading removal from SQLite, Qdrant vectors, and Neo4j graph nodes.

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

#if you get a script execution policy error,
#First run this once: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

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

> [!TIP]
> **SSL Certificate / Firewall Issue (`+s` vs `+ssc`):**
> If you get `[SSL: CERTIFICATE_VERIFY_FAILED]` or `Unable to retrieve routing information` (common on college/office Wi-Fi, VPNs, or antivirus software with HTTPS scanning), change `neo4j+s://` to **`neo4j+ssc://`** in your `.env`:
> ```env
> NEO4J_URI=neo4j+ssc://<your-instance-id>.databases.neo4j.io
> ```
> `+ssc` enables encryption while accepting self-signed or proxy-intercepted certificates in the network chain.

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
|---|---|---|
| `GET` | `/health` | Service connectivity status (AuraDB, Qdrant, SQLite, LLM) |
| `POST` | `/upload` | Upload one or multiple PDF, TXT, or MD files for background ingestion |
| `GET` | `/documents` | List all uploaded documents with status and chunk counts |
| `GET` | `/documents/{id}` | Get document metadata |
| `GET` | `/documents/{id}/status` | Poll ingestion & extraction chunk progress |
| `GET` | `/documents/{id}/chunks` | Inspect document chunks and page mappings |
| `DELETE` | `/documents/{id}` | Cascade delete document, vectors, and graph entities |
| `POST` | `/query` | Ask questions in `graph` or `vector` mode |
| `GET` | `/graph/summary` | Knowledge graph entity and relationship counts |
| `GET` | `/graph/document/{id}` | Subgraph nodes and edges for a specific document |
| `GET` | `/graph/global` | Entire connected knowledge graph across all uploaded documents |
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

---

## ❓ Troubleshooting

### Neo4j AuraDB Connection Issues

| Issue | Cause | Solution |
|---|---|---|
| `[SSL: CERTIFICATE_VERIFY_FAILED]` or `self-signed certificate in certificate chain` | Antivirus (Bitdefender, Kaspersky, Avast) or institutional Wi-Fi intercepting SSL certificates with local certs. | Change `neo4j+s://` to **`neo4j+ssc://`** in `.env`. |
| `Unable to retrieve routing information` | Port 7687 blocked by firewall/VPN or AuraDB instance is paused. | 1. Resume instance in [Aura Console](https://console.neo4j.io/).<br>2. Try `NEO4J_URI=bolt+ssc://<id>.databases.neo4j.io:7687`. |
| `ModuleNotFoundError: No module named '...'` | Virtual environment not activated or packages not installed. | Run `.venv\Scripts\activate` then `pip install -r requirements.txt`. |

