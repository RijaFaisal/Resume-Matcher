import pickle
from pathlib import Path

meta_path = Path("./vectorstore/metadata.pkl")
with meta_path.open("rb") as f:
    metadata = pickle.load(f)

print(f"Total chunks: {len(metadata)}")
print(metadata[:5])  # show first 5 chunks