from src.api.main import MatchRequest, MatchResponse, match_resume

SAMPLE_RESUME = "Experienced data scientist with Python, ML, and SQL expertise. Looking for new challenges."


def test_match_resume_logic():
    class DummyRequest:
        app = type("App", (), {"state": match_resume.__globals__["app"].state})()

    req = MatchRequest(resume_text=SAMPLE_RESUME, top_n=2)
    response = match_resume(req, DummyRequest())
    assert isinstance(response, MatchResponse)
    assert len(response.matches) == 2
    for m in response.matches:
        assert hasattr(m, "rank")
        assert hasattr(m, "job_title")
        assert hasattr(m, "similarity_score")
