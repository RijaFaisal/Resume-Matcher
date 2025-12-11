# scripts/ingest_faiss.py
import os
import argparse
import json
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def load_docs_from_local(path):
    path = Path(path)
    docs = []
    if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
        df = pd.read_csv(path)
        # expect a text column named 'text' or 'content' or 'job_description'
        for col in ("text","content","job_description","job_title","job"):
            if col in df.columns:
                texts = df[col].astype(str).tolist()
                break
        else:
            texts = df.iloc[:,0].astype(str).tolist()
        docs = [{"id": str(i), "text": t} for i,t in enumerate(texts)]
    else:
        # walk txt files
        for f in path.rglob("*.txt"):
            docs.append({"id": str(len(docs)), "text": f.read_text(encoding="utf-8")})
    return docs

def load_docs_from_s3(s3_uri, aws_region=None):
    # s3_uri e.g. s3://bucket/path/
    s3 = boto3.client("s3", region_name=aws_region)
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, prefix = bucket_key.partition("/")
    objs = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    docs = []
    for obj in objs.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        resp = s3.get_object(Bucket=bucket, Key=key)
        txt = resp["Body"].read().decode("utf-8")
        docs.append({"id": key, "text": txt})
    return docs

def embed_and_index(docs, model_name="sentence-transformers/all-MiniLM-L6-v2", dim=None):
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if dim is None:
        dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner-product on normalized vectors => cosine if normalized
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings

def save_index_and_metadata(index, docs, embeddings, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / "faiss_index.bin"
    meta_path = out_dir / "meta.json"
    emb_path = out_dir / "embeddings.npy"
    faiss.write_index(index, str(idx_path))
    np.save(str(emb_path), embeddings)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return idx_path, meta_path, emb_path

def upload_to_s3(files, s3_uri, aws_region=None):
    s3 = boto3.client("s3", region_name=aws_region)
    _, _, bucket_key = s3_uri.partition("s3://")
    bucket, _, prefix = bucket_key.partition("/")
    for f in files:
        key = (prefix.rstrip("/") + "/" + Path(f).name).lstrip("/")
        s3.upload_file(str(f), bucket, key)
        print(f"Uploaded {f} -> s3://{bucket}/{key}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True,
                   help="Local path or s3://bucket/prefix to load docs")
    p.add_argument("--out-dir", default="vectorstore/", help="local output dir")
    p.add_argument("--s3-dest", default=None, help="s3://bucket/path to upload index")
    p.add_argument("--embed-model", default=os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
    p.add_argument("--aws-region", default=os.getenv("AWS_REGION"))
    args = p.parse_args()

    if args.source.startswith("s3://"):
        docs = load_docs_from_s3(args.source, aws_region=args.aws_region)
    else:
        docs = load_docs_from_local(args.source)

    if not docs:
        print("No documents found at source:", args.source)
        return

    index, embeddings = embed_and_index(docs, model_name=args.embed_model)
    idx_path, meta_path, emb_path = save_index_and_metadata(index, docs, embeddings, args.out_dir)

    print("Saved index & metadata locally:", idx_path, meta_path, emb_path)

    if args.s3_dest:
        upload_to_s3([idx_path, meta_path, emb_path], args.s3_dest, aws_region=args.aws_region)

if __name__ == "__main__":
    main()
