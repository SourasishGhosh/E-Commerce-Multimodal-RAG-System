import json
from datetime import datetime, UTC
from pathlib import Path

LOG_FILE = Path("data/feedback_log.jsonl")

def log_click(query_id: str, product_id: str, query_vector: list[float]):
   
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query_id": query_id,
        "product_id": product_id,
        "query_vector": query_vector,   # stored for offline reranking
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")