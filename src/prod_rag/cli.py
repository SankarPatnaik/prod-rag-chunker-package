from __future__ import annotations

import json

import typer

from prod_rag.models import AppConfig
from prod_rag.pipeline.rag_pipeline import RAGPipeline
from prod_rag.utils.io import read_yaml

app = typer.Typer(help="Production RAG CLI")


def load_config(config_path: str) -> AppConfig:
    raw = read_yaml(config_path)
    return AppConfig(**raw)


@app.command()
def index(config: str, path: str):
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    stats = pipeline.index_document(path)
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def ask(config: str, query: str):
    cfg = load_config(config)
    pipeline = RAGPipeline(cfg)
    answer = pipeline.ask(query)
    typer.echo(json.dumps(answer.model_dump(), ensure_ascii=False, indent=2))


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run("prod_rag.service.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
