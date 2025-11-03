import requests
import sys


def main(project_id: str) -> None:
    payload = {
        "name": "Resume NLP Monitoring",
        "description": "NLP metrics (OOV, length, etc.) and data drift for resumes",
        "dashboard": {
            "name": "Resume NLP Monitoring",
            "panels": [
                {"title": "Text Overview", "filter": {"metadata_values": {}, "tag_values": []}, "size": 2},
                {"title": "Data Drift", "filter": {"metadata_values": {}, "tag_values": []}, "size": 2},
            ],
        },
    }
    url = f"http://localhost:7000/api/projects/{project_id}/info"
    r = requests.post(url, json=payload, timeout=30)
    print(r.status_code)
    print((r.text or "").strip())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: update_dashboard.py <project_id>")
        sys.exit(1)
    main(sys.argv[1])


