import sys
<<<<<<< HEAD
import requests
from pathlib import Path

=======
from pathlib import Path

import requests

>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a

def main(project_id: str, snapshot_path: str) -> None:
    path = Path(snapshot_path)
    raw = path.read_text(encoding="utf-8")
    url = f"http://localhost:7000/api/projects/{project_id}/snapshots"
<<<<<<< HEAD
    resp = requests.post(url, data=raw.encode("utf-8"), headers={"Content-Type": "application/json"}, timeout=60)
=======
    resp = requests.post(
        url,
        data=raw.encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
    print(resp.status_code)
    print((resp.text or "").strip()[:500])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: post_snapshot.py <project_id> <snapshot_json_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
<<<<<<< HEAD


=======
>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
