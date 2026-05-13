import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from src.core.config import settings

client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
SCORE_FILE = Path("data/rerank_scores.json")

def load_rerank_scores() -> dict:
    if SCORE_FILE.exists():
        with open(SCORE_FILE, "r") as f:
            return json.load(f)
    return {}

def build_qdrant_filter(parsed_filters: dict) -> Filter | None:
    pass 

def search(query_vector: list[float], parsed_filters: dict, top_k: int = 10):
    qdrant_filter = build_qdrant_filter(parsed_filters)
    
    
    fetch_k = top_k * 2 
    
    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        query_filter=qdrant_filter,
        limit=fetch_k,
        with_payload=True
    )
    
    rerank_scores = load_rerank_scores()
    
    final_results = []
    for r in results:
        base_score = r.score
        boost = rerank_scores.get(str(r.id), 0.0) * 0.10 
        final_score = base_score + boost
        
        final_results.append({
            "id": r.id, 
            "original_score": base_score,
            "final_score": final_score, 
            "payload": r.payload
        })
    
    # Sort by the new final score and return the requested top_k
    final_results.sort(key=lambda x: x["final_score"], reverse=True)
    return final_results[:top_k]