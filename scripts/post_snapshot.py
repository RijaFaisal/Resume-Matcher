import sys
import requests
from pathlib import Path


def main(project_id: str, snapshot_path: str) -> None:
    path = Path(snapshot_path)
    raw = path.read_text(encoding="utf-8")
    url = f"http://localhost:7000/api/projects/{project_id}/snapshots"
    resp = requests.post(url, data=raw.encode("utf-8"), headers={"Content-Type": "application/json"}, timeout=60)
    print(resp.status_code)
    print((resp.text or "").strip()[:500])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: post_snapshot.py <project_id> <snapshot_json_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])


