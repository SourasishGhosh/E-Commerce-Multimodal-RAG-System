import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_search_missing_inputs():
    """API should return 400 if neither text nor image is provided."""
    response = client.post("/api/v1/search", data={"top_k": 5})
    assert response.status_code == 400
    assert "Provide text query, image, or both" in response.json()["detail"]

@patch("src.api.routes.text_embedder")
@patch("src.api.routes.search")
def test_search_text_only(mock_search, mock_embedder):
    """Test standard text search routing."""
    # Mock the dependencies
    mock_embedder.embed_text.return_value = [0.1] * 512
    mock_search.return_value = [{"id": "1", "final_score": 0.9, "payload": {}}]
    
    response = client.post("/api/v1/search", data={"query": "wooden table", "top_k": 5})
    
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1