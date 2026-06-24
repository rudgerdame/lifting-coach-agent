"""Chunk markdown corpus, embed, and build a FAISS index."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR = Path("index/corpus")
STORE_DIR = Path("index/store")
INDEX_PATH = STORE_DIR / "faiss.index"
CHUNKS_PATH = STORE_DIR / "chunks.jsonl"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    text: str


def _parse_source(lines: list[str], filename: str) -> str:
    for line in lines[:5]:
        match = re.match(r"^source:\s*(.+)$", line.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return filename


def _chunk_markdown(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    source = _parse_source(lines, path.name)

    sections: list[tuple[str, list[str]]] = []
    current_title = path.stem.replace("_", " ").title()
    current_body: list[str] = []

    for line in lines:
        if line.startswith("# "):
            if current_body:
                sections.append((current_title, current_body))
            current_title = line[2:].strip()
            current_body = []
            continue
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, current_body))
            current_title = line[3:].strip()
            current_body = []
            continue
        current_body.append(line)

    if current_body:
        sections.append((current_title, current_body))

    chunks: list[Chunk] = []
    for i, (title, body_lines) in enumerate(sections):
        body = "\n".join(body_lines).strip()
        if len(body) < 40:
            continue
        chunk_id = f"{path.stem}:{i}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                source=source,
                title=title,
                text=f"{title}\n\n{body}",
            )
        )
    return chunks


def load_corpus_chunks(corpus_dir: Path = CORPUS_DIR) -> list[Chunk]:
    paths = sorted(corpus_dir.glob("*.md"))
    if not paths:
        raise FileNotFoundError(f"No markdown files in {corpus_dir}")

    chunks: list[Chunk] = []
    for path in paths:
        chunks.extend(_chunk_markdown(path))
    return chunks


def build_index(
    corpus_dir: Path = CORPUS_DIR,
    store_dir: Path = STORE_DIR,
    model_name: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    chunks = load_corpus_chunks(corpus_dir)
    model = SentenceTransformer(model_name)
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "title": chunk.title,
                        "text": chunk.text,
                    }
                )
                + "\n"
            )

    meta = {
        "model_name": model_name,
        "n_chunks": len(chunks),
        "dim": vectors.shape[1],
    }
    (store_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return INDEX_PATH, CHUNKS_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from index/corpus/")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--store-dir", type=Path, default=STORE_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    index_path, chunks_path = build_index(
        corpus_dir=args.corpus_dir,
        store_dir=args.store_dir,
        model_name=args.model,
    )
    print(f"Wrote {index_path}")
    print(f"Wrote {chunks_path}")


if __name__ == "__main__":
    main()
