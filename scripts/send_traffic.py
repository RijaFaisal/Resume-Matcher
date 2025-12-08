import time
<<<<<<< HEAD
=======

>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
import requests


def main(num_requests: int = 30, delay_seconds: float = 0.2) -> None:
    payload = {"resume_text": "A" * 800, "top_n": 3}
    url = "http://localhost:8000/match_resume"
    for _ in range(num_requests):
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception:
            pass
        time.sleep(delay_seconds)


if __name__ == "__main__":
    main()
<<<<<<< HEAD


=======
>>>>>>> ad96fb2ff61387c387f69110253228d7040afb5a
