from __future__ import annotations

import os
import time

import chromadb
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pydantic import BaseModel

VECTOR_DIM = 384

CHROMA_DIR = os.environ.get("BASELINE_PERSIST_DIR", "/data/chroma")
MODEL_NAME = os.environ.get("BASELINE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class EmbedRequest(BaseModel):
    texts: list[str]


class IngestItem(BaseModel):
    id: str
    text: str


class IngestRequest(BaseModel):
    items: list[IngestItem]


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


def create_app(persist_dir: str | None = None, model_name: str | None = None) -> FastAPI:
    _persist_dir = persist_dir or CHROMA_DIR
    _model_name = model_name or MODEL_NAME

    app = FastAPI()

    _model: SentenceTransformerEmbeddings | None = None
    _collection: chromadb.Collection | None = None

    def _get_model() -> SentenceTransformerEmbeddings:
        nonlocal _model
        if _model is None:
            _model = SentenceTransformerEmbeddings(
                model_name=_model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        return _model

    def _embed(texts: list[str]) -> list[list[float]]:
        return _get_model().embed_documents(texts)

    def _get_collection() -> chromadb.Collection:
        nonlocal _collection
        if _collection is None:
            client = chromadb.PersistentClient(path=_persist_dir)
            _collection = client.get_or_create_collection(
                name="vectors",
                metadata={"hnsw:space": "cosine"},
            )
        return _collection

    def _json_err(status: int, msg: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": msg})

    @app.get("/v1/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/stats")
    async def stats():
        _ = _get_model()
        collection = _get_collection()
        count = collection.count()
        return {
            "rows": count,
            "index": "flat",
            "model": _model_name,
            "dim": VECTOR_DIM,
        }

    @app.post("/v1/embed")
    async def embed(req: EmbedRequest):
        if not req.texts:
            return _json_err(400, "texts must be non-empty")
        if len(req.texts) > 256:
            return _json_err(400, "texts batch exceeds 256")
        t0 = time.perf_counter()
        embeddings = _embed(req.texts)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "embeddings": embeddings,
            "model": _model_name,
            "dim": VECTOR_DIM,
            "latency_ms": latency_ms,
        }

    @app.post("/v1/ingest")
    async def ingest(req: IngestRequest):
        if not req.items:
            return _json_err(400, "items must be non-empty")
        collection = _get_collection()
        texts = [item.text for item in req.items]
        embeddings = _embed(texts)
        collection.upsert(
            ids=[item.id for item in req.items],
            documents=texts,
            embeddings=embeddings,
        )
        return {"ingested": len(req.items)}

    @app.post("/v1/search")
    async def search(req: SearchRequest):
        if not req.query.strip():
            return _json_err(400, "query must be non-empty")
        top_k = max(1, min(1000, req.top_k or 10))
        collection = _get_collection()
        query_embedding = _embed([req.query])
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "distances"],
        )
        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                score = 1.0 - distance
                text = results["documents"][0][i] if results["documents"] else ""
                hits.append({
                    "id": doc_id,
                    "text": text,
                    "score": score,
                    "distance": distance,
                })
        return {"results": hits}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8081)
