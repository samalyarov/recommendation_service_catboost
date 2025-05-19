import pytest
from fastapi import status

def test_recommendations_endpoint_success(client, test_user_data, test_post_data):
    """Test successful recommendation request."""
    response = client.get("/post/recommendations/?id=1&limit=5")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "exp_group" in data
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 5

def test_recommendations_endpoint_invalid_user(client):
    """Test recommendation request with non-existent user."""
    response = client.get("/post/recommendations/?id=999999&limit=5")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "User not found" in response.json()["detail"]

def test_recommendations_endpoint_invalid_limit(client):
    """Test recommendation request with invalid limit."""
    response = client.get("/post/recommendations/?id=1&limit=0")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_recommendations_endpoint_missing_id(client):
    """Test recommendation request without user ID."""
    response = client.get("/post/recommendations/?limit=5")
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_recommendations_endpoint_default_limit(client):
    """Test recommendation request with default limit."""
    response = client.get("/post/recommendations/?id=1")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert len(data["recommendations"]) <= 5  # Default limit

def test_recommendations_response_structure(client):
    """Test that response has correct structure."""
    response = client.get("/post/recommendations/?id=1&limit=1")
    assert response.status_code == status.HTTP_200_OK
    
    data = response.json()
    assert "exp_group" in data
    assert "recommendations" in data
    
    if data["recommendations"]:
        recommendation = data["recommendations"][0]
        assert "id" in recommendation
        assert "text" in recommendation
        assert "topic" in recommendation 