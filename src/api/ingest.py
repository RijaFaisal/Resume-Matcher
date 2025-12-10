"""
src/ingest.py

Simple, modular ingestion pipeline:
- Reads .txt and .pdf files from ./data/
- Splits text into overlapping chunks
- Generates embeddings via sentence-transformers
- Stores vectors in FAISS and metadata in ./vectorstore/

Usage (example):
    python -m src.ingest --data_dir ./data --index_dir ./vectorstore --model all-MiniLM-L6-v2
"""

import argparse
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import PyPDF2

# ---------- Helpers ----------


def read_txt(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(file_path: Path) -> str:
    text = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            try:
                page_text = p.extract_text() or ""
            except Exception:
                page_text = ""
            text.append(page_text)
    return "\n".join(text)


def list_files(data_dir: Path, exts=(".txt", ".pdf")) -> List[Path]:
    files = []
    for ext in exts:
        files.extend(sorted(data_dir.rglob(f"*{ext}")))
    return files


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """
    Very simple chunking by characters. Overlap ensures context continuity.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text = text.replace("\r\n", "\n")
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)  # ensure progress
    return [c for c in chunks if c]


# ---------- Ingest pipeline ----------


class Ingestor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"[Ingestor] loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        # returns a 2D numpy array of shape (n_texts, d)
        embeddings = self.embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )
        return np.array(embeddings).astype("float32")

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)  # use inner product; we'll normalize
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index

    def save_index(self, index: faiss.Index, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        idx_path = index_dir / "faiss.index"
        faiss.write_index(index, str(idx_path))
        print(f"[Ingestor] FAISS index written to {idx_path}")

    def save_metadata(self, metas: List[Dict], index_dir: Path) -> None:
        meta_path = index_dir / "metadata.pkl"
        with meta_path.open("wb") as f:
            pickle.dump(metas, f)
        print(f"[Ingestor] metadata saved to {meta_path}")

    def ingest(
        self, data_dir: Path, index_dir: Path, chunk_size: int = 800, overlap: int = 200
    ) -> None:
        files = list_files(data_dir)
        print(f"[Ingestor] found {len(files)} files in {data_dir}")
        all_chunks = []
        metas = []
        for file in files:
            ext = file.suffix.lower()
            if ext == ".txt":
                text = read_txt(file)
            elif ext == ".pdf":
                text = read_pdf(file)
            else:
                print(f"[Ingestor] skipping unsupported file {file}")
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for i, c in enumerate(chunks):
                metas.append(
                    {"source": str(file), "chunk_id": i, "text_snippet": c[:200]}
                )
                all_chunks.append(c)
        if not all_chunks:
            raise RuntimeError(
                "No chunks generated. Check your data directory and supported file types."
            )

        print(f"[Ingestor] total chunks: {len(all_chunks)}")
        embeddings = self.embed_texts(all_chunks)
        # normalize embeddings for inner product search
        faiss.normalize_L2(embeddings)
        index = self.build_faiss_index(embeddings)
        self.save_index(index, index_dir)
        self.save_metadata(metas, index_dir)
        # also save embedding model name
        model_path = index_dir / "model_name.txt"
        model_path.write_text(self.model_name)
        print("[Ingestor] ingestion complete.")


# ---------- CLI ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory with .txt and .pdf files",
    )
    p.add_argument(
        "--index_dir",
        type=str,
        default="./vectorstore",
        help="Directory to save FAISS index and metadata",
    )
    p.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    p.add_argument("--chunk_size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    index_dir = Path(args.index_dir)
    ing = Ingestor(model_name=args.model)
    ing.ingest(
        data_dir=data_dir,
        index_dir=index_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
