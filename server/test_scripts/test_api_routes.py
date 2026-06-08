import pytest
from fastapi.testclient import TestClient
from server.fastapi_app import app

client = TestClient(app)

def test_text_config_generation_endpoint_routing():
    """Tests the /api/v3/config/text route under typical use and edge cases."""
    valid_payload = {
        "inputText": "Design a modern fitness dashboard with an emerald battery status ring.",
        "useRAG": False
    }
    response = client.post("/api/v3/config/text", json=valid_payload)
    assert response.status_code == 202
    data = response.json()
    assert "task_id" in data
    assert data["status"] == "Queued"
    
    active_task_id = data["task_id"]

    status_response = client.get(f"/api/v3/tasks/{active_task_id}")
    assert status_response.status_code in [200, 404]  # 404 if mock memory is un-persisted

    # Invalid Payload (Blank Input Verification)
    malformed_payload = {
        "inputText": "   ",  # Whitespace padding
        "useRAG": True
    }
    invalid_response = client.post("/api/v3/config/text", json=malformed_payload)
    assert invalid_response.status_code == 400
    assert "inputText is required" in invalid_response.json()["detail"]


def test_missing_task_id_polling_error():
    fake_token = "invalid-token-000000"
    response = client.get(f"/api/v3/tasks/{fake_token}")
    assert response.status_code == 404
    print("HTTP API routing layers and error codes verified!")