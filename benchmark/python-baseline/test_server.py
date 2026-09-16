import tempfile
import os

from fastapi.testclient import TestClient
from server import create_app


def test_health_and_search():
    with tempfile.TemporaryDirectory() as td:
        client = TestClient(
            create_app(
                persist_dir=td,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        assert client.get("/v1/health").status_code == 200
        r = client.post("/v1/embed", json={"texts": ["hello world"]})
        assert r.status_code == 200
        assert len(r.json()["embeddings"][0]) == 384

        r = client.post(
            "/v1/ingest",
            json={"items": [{"id": "1", "text": "the cat sits outside"}]},
        )
        assert r.status_code == 200

        r = client.post("/v1/search", json={"query": "cat outside", "top_k": 5})
        assert r.status_code == 200
        hits = r.json()["results"]
        assert hits and hits[0]["id"] == "1"


def test_empty_embed_is_400():
    with tempfile.TemporaryDirectory() as td:
        client = TestClient(
            create_app(
                persist_dir=td,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        assert client.post("/v1/embed", json={"texts": []}).status_code == 400
