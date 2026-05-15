# E-Commerce Multi-Modal RAG System

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-OpenCLIP-EE4C2C.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) system for e-commerce. This engine sums up text and image data into a single vector space, enabling users to search catalogs using natural language, image uploads, or both simultaneously—without the hallucination risks of traditional generative search.

[Watch the Demonstration Video](E-Commerce-Multimodal-RAG-System/assets/)



https://github.com/user-attachments/assets/59a43ebc-77c7-49f7-9895-fdd06610a470


---

## Architecture & Engineering Highlights

- **Multi-Modal Shared Space:** Utilizes `OpenCLIP` (PyTorch) to map both product images and text descriptions into the same 512-dimensional vector space, enabling complex cross-modal search.
- **Pre-ANN Semantic Filtering:** Eliminates hallucinated recommendations by using a lightweight LLM during the query phase to extract strict constraints (e.g., `price < 400`). These are applied natively in Qdrant *before* the Nearest Neighbor search.
- **Safe Offline Reranking:** Implements an asynchronous `/click` feedback loop. User interactions update an offline popularity score applied post-retrieval, avoiding the database-corrupting anti-pattern of mutating production embeddings in real-time.

---

## Architectural Blueprint

### Architectural Pattern

- **Multi-Modal Retrieval-Augmented Generation (RAG):** Combines multi-modal vector search capabilities (processing text queries, images, or both simultaneously) with a generative large language model to create a context-aware conversational shopping assistant.
- **Semantic Vector Space Alignment:** Maps images and text into a shared mathematical space, allowing the system to match an abstract text concept (e.g., "comfortable running shorts") directly to product images and metadata.

---

## Technology Stack

### Backend Layer


| **Component**   | **Technology** | **Purpose**                                   |
| --------------- | -------------- | --------------------------------------------- |
| Core Framework  | Python 3       | Main programming language powering the logic. |
| API Engine      | FastAPI        | High-performance, asynchronous web framework. |
| Server Gateway  | Uvicorn        | ASGI server implementation to run FastAPI.    |
| Data Validation | Pydantic       | Validates incoming JSON data structures.      |
| Rate Limiter    | SlowAPI        | Protects endpoints from abuse.                |


### Frontend Layer

- **Markup:** HTML5
- **Styling:** Tailwind CSS (via CDN)
- **Client Logic:** Vanilla JavaScript

### AI & Vector Engines

- **Embedding Generator:** OpenCLIP (`CLIPEmbedder`, `CLIPTextEmbedder`)
- **Generative AI Layer:** Google Gemini API (`gemini-2.5-flash`)
- **Vector Database:** Qdrant (`qdrant/qdrant:latest`)

### DevOps & Infrastructure

- **Containerization:** Docker, Docker Compose
- **Cloud Host:** Microsoft Azure (Ubuntu Server VM)
- **Networking:** NSGs with inbound firewall rules for TCP port 8000
- **GPU Acceleration:** NVIDIA Docker Container Toolkit (RTX 3050)

### Data Engineering

- **Version Control:** Git & GitHub
- **Transport Utilities:** SCP, SSH (private key authentication)
- **Environment Security:** Dotenv (`.env`)

---

## Repository Structure

```text
E-Commerce-Multimodal-RAG-System/  
├── assets/                  
├── src/
│   ├── api/                   # FastAPI routing and Pydantic schemas
│   ├── core/                  # configs and logging setup
│   ├── ingestion/             # Batch ingestion and metadata extraction
│   ├── models/                # PyTorch/CLIP wrappers for text and image
│   ├── retrieval/             # Pre-ANN filtering and Qdrant search logic
│   ├── feedback/              # Click logging and offline nightly reranker
│   └── main.py                # ASGI application entry point
├── tests/                     # Pytest suite 
├── docker-compose.yml         
├── Dockerfile                 
├── requirements.txt           
└── README.md           
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/SourasishGhosh/E-Commerce-Multimodal-RAG-System.git
cd E-Commerce-Multimodal-RAG-System
```

### 2. Configure environment

* E-Commerce Multi-Modal RAG System - Environment Configuration Template
Copy this file to a new file named '.env' and fill in your actual credentials.
DO NOT commit the actual '.env' file to your version control repository (GitHub).

* Google Gemini API Settings
Get your API key from Google AI Studio (https://aistudio.google.com/)

`note` : You can use any API service (e.g.: OpenAI, Claude etc.)

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```
* Qdrant Vector Database Settings
 Use 'qdrant' if running inside Docker network, or 'localhost' if running API locally
```bash
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=products
```
* Application Server Settings

```bash
APP_HOST=0.0.0.0
APP_PORT=8000
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 3. Start the infrastructure

```bash
docker-compose up -d --build
```
*The API will be available at `http://localhost:8000`*

### 4. Ingest sample data

```bash
docker exec -it multimodal_api python -m src.ingestion.load_qdrant
```

---

## Testing

Run the test suite to ensure the vector space and routing logic are sound:

```bash
docker exec -it multimodal_api pytest tests/ -v
```

---

## API Usage

### `POST /api/v1/search`

Perform a multi-modal semantic search. Accepts `multipart/form-data`.

**Parameters:**

- `query` (string, optional): Text description with implicit filters (e.g., "Spongebob pants under $30")
- `image` (file, optional): An image to base the visual search on.
- `top_k` (int, optional): Number of results to return (default: 10).

**Example Response:**

```json
{
  "results": [
    {
      "id": "uuid-1234",
      "original_score": 0.89,
      "final_score": 0.94,
      "payload": {
        "product_name": "Eames Lounge Chair Replica",
        "price": 199.99,
        "style": "mid-century",
        "material": "leather"
      }
    }
  ],
  "filters_applied": {
    "price_max": 200,
    "style": "mid-century"
  }
}
```

### `POST /api/v1/click`

Log a user interaction for the offline reranking pipeline.

**Payload:**

```json
{
  "query_id": "session-xyz",
  "product_id": "uuid-1234",
  "query_vector": [0.012, -0.045, ...]
}
```

* For further queries contact:  sourasishghosh02@gmail.com
