import pytest
from unittest.mock import patch, MagicMock
from src.retrieval.search_engine import build_qdrant_filter, search

def test_build_qdrant_filter_empty():
    """Test that an empty parsed dict returns no filter."""
    assert build_qdrant_filter({}) is None

def test_build_qdrant_filter_with_constraints():
    """Test that parsed constraints are properly mapped to Qdrant condition objects."""
    filters = {"price_max": 200, "style": "industrial"}
    q_filter = build_qdrant_filter(filters)
    
    assert q_filter is not None
    assert len(q_filter.must) == 2
    
    # Check if 'style' is in the must conditions
    keys = [condition.key for condition in q_filter.must]
    assert "style" in keys
    assert "price" in keys

@patch("src.retrieval.search_engine.client")
def test_search_applies_rerank(mock_qdrant_client):
    """Test that the search function correctly applies offline rerank scores."""
    # Mock Qdrant response
    mock_result = MagicMock()
    mock_result.id = "prod_123"
    mock_result.score = 0.85
    mock_result.payload = {"name": "Test Table"}
    mock_qdrant_client.search.return_value = [mock_result]
    
    # Mock the rerank scores loading
    with patch("src.retrieval.search_engine.load_rerank_scores") as mock_scores:
        mock_scores.return_value = {"prod_123": 1.0} # Max popularity boost
        
        # Execute search with dummy vector
        results = search([0.1] * 512, parsed_filters={})
        
        assert len(results) == 1
        assert results[0]["id"] == "prod_123"
        # 0.85 base score + (1.0 * 0.10 max boost) = 0.95
        assert round(results[0]["final_score"], 2) == 0.95