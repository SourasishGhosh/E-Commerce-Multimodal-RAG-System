
#  Conversational Product Discovery Engine

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-FF5252.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-OpenCLIP-EE4C2C.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) system for e-commerce. This engine unifies text and image data into a single vector space, allowing users to search catalogs using natural language, image uploads, or both simultaneously without the hallucination risks of traditional generative search.

---

##  Architecture & Engineering Highlights

* **Multi-Modal Shared Space:** Utilizes `OpenCLIP` (PyTorch) to map both product images and text descriptions into the exact same 512-dimensional vector space, enabling complex cross-modal search.
* **Pre-ANN Semantic Filtering:** Eliminates hallucinated recommendations by using a lightweight LLM (GPT-4o-mini) during the query phase to extract strict constraints (e.g., `price < 400`). These are applied natively in Qdrant *before* the Nearest Neighbor search.
* **Safe Offline Reranking:** Implements an asynchronous `/click` feedback loop. User interactions update an offline popularity score applied post-retrieval, avoiding the database-corrupting anti-pattern of mutating production embeddings in real-time.

---

##  Tech Stack

* **Backend:** FastAPI, Python 3.11, Pydantic
* **Vector DB:** Qdrant (Local via Docker / Scalable to Qdrant Cloud)
* **AI/ML:** PyTorch, OpenCLIP, OpenAI (for deterministic metadata extraction)
* **Data Processing:** Pandas
* **Deployment & CI/CD:** Docker, Docker Compose, Pytest

---

##  Repository Structure

```text
ecommerce-multimodal-rag/
├── data/                      # Raw catalogs and local images
├── notebooks/                 # Jupyter notebooks for embedding prototyping
├── src/
│   ├── api/                   # FastAPI routing and Pydantic schemas
│   ├── core/                  # Environment configs and logging setup
│   ├── ingestion/             # Batch ingestion and LLM metadata extraction
│   ├── models/                # PyTorch/CLIP wrappers for text and image
│   ├── retrieval/             # Pre-ANN filtering and Qdrant search logic
│   ├── feedback/              # Click logging and offline nightly reranker
│   └── main.py                # ASGI application entry point
├── tests/                     # Pytest suite (Unit & Integration)
├── docker-compose.yml         # Container orchestration
├── Dockerfile                 # Application container
├── requirements.txt           # Python dependencies
└── .env.example               # Template for environment secrets

```


##  Setup & Installation

### 1. Clone the repository and configure environment

```bash
git clone [https://github.com/yourusername/ecommerce-multimodal-rag.git](https://github.com/yourusername/ecommerce-multimodal-rag.git)
cd ecommerce-multimodal-rag

# Copy the environment template and add your OpenAI API Key
cp .env.example .env

```

### 2. Start the infrastructure

Spin up the FastAPI application and the local Qdrant vector database.

```bash
docker-compose up -d --build

```

*The API will be available at `http://localhost:8000*`

### 3. Ingest sample data

Run the offline batch processor to extract metadata, generate OpenCLIP embeddings, and push them to Qdrant.

```bash
docker exec -it <container_name> python -m src.ingestion.load_qdrant

```

---

##  Testing

The project maintains rigorous unit and integration tests using `pytest` and `FastAPI.testclient`. Run the suite to ensure the vector space and routing logic are sound.

```bash
docker exec -it <container_name> pytest tests/ -v

```

---

##  API Usage

### `POST /api/v1/search`

Perform a multi-modal semantic search. Accepts `multipart/form-data`.

**Parameters:**

* `query` (string, optional): Text description with implicit filters (e.g., "mid-century chair under $200")
* `image` (file, optional): An image to base the visual search on.
* `top_k` (int, optional): Number of results to return (default: 10).

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

