# tests/test_model.py
from src.api.main import MatchRequest, match_resume, MatchResponse

SAMPLE_RESUME = "Experienced data scientist with Python, ML, and SQL expertise. Looking for new challenges."


def test_match_resume_logic():
    # Create dummy request object
    class DummyRequest:
        app = type("App", (), {"state": match_resume.__globals__["app"].state})()

    req = MatchRequest(resume_text=SAMPLE_RESUME, top_n=2)
    response = match_resume(req, DummyRequest())
    assert isinstance(response, MatchResponse)
    assert len(response.matches) == 2
    assert all(
        hasattr(m, "rank")
        and hasattr(m, "job_title")
        and hasattr(m, "similarity_score")
        for m in response.matches
    )
