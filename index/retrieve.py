"""Search the methodology corpus FAISS index."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from index.build import CHUNKS_PATH, DEFAULT_MODEL, INDEX_PATH, STORE_DIR


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    source: str
    title: str
    text: str
    score: float


class CorpusRetriever:
    def __init__(
        self,
        store_dir: Path = STORE_DIR,
        model_name: str | None = None,
    ) -> None:
        index_path = store_dir / "faiss.index"
        chunks_path = store_dir / "chunks.jsonl"
        meta_path = store_dir / "meta.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Missing index at {store_dir}. Run: python -m index.build"
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.model_name = model_name or meta.get("model_name", DEFAULT_MODEL)
        self.model = SentenceTransformer(self.model_name)
        self.index = faiss.read_index(str(index_path))
        self.chunks: list[dict] = [
            json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    def search(self, query: str, k: int = 3) -> list[RetrievalHit]:
        vector = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(np.asarray(vector, dtype=np.float32), k)

        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.chunks[int(idx)]
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    source=row["source"],
                    title=row["title"],
                    text=row["text"],
                    score=float(score),
                )
            )
        return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve coaching corpus chunks")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to return")
    parser.add_argument("--store-dir", type=Path, default=STORE_DIR)
    args = parser.parse_args()

    retriever = CorpusRetriever(store_dir=args.store_dir)
    hits = retriever.search(args.query, k=args.k)
    for i, hit in enumerate(hits, start=1):
        print(f"--- Hit {i} (score={hit.score:.3f}) [{hit.source}] ---")
        print(hit.text)
        print()


if __name__ == "__main__":
    main()
