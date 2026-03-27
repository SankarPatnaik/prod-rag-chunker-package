from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from prod_rag.service.api import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "prod-rag-chunker API is running"
    assert data["health"] == "/health"
    assert data["docs"] == "/docs"


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chunk_endpoint_with_inline_text():
    payload = {
        "document_id": "inline-1",
        "text": "ASSETS\n\nCash balance was 2M USD in 2025.",
        "strategy": "section_aware",
    }
    response = client.post("/v1/chunk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["document_id"] == "inline-1"
    assert data["stats"]["num_child_chunks"] >= 1
