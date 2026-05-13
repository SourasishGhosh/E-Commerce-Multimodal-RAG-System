from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class SearchResult(BaseModel):
    id: str
    original_score: float
    final_score: float
    payload: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    filters_applied: Dict[str, Any]

class ClickEventRequest(BaseModel):
    query_id: str = Field(..., description="Unique ID for the search session")
    product_id: str = Field(..., description="The ID of the clicked product")
    query_vector: List[float] = Field(..., description="The 512-dim vector of the query")