from src.rag.prompts import get_chat_prompt, SYSTEM_ROLE, PROMPT_INSTRUCTION


def test_get_chat_prompt_structure():
    """Verify prompt prompt contains all expected components."""
    context = "Doc A"
    user_context = "User Profile X"
    query = "Hello"

    prompt = get_chat_prompt(
        SYSTEM_ROLE, PROMPT_INSTRUCTION, context, user_context, query
    )

    assert SYSTEM_ROLE in prompt
    assert PROMPT_INSTRUCTION in prompt
    assert "Doc A" in prompt
    assert "User Profile X" in prompt
    assert "Hello" in prompt


def test_get_chat_prompt_no_user_context():
    """Verify prompt handles missing user context correctly."""
    context = "Doc B"
    query = "Query Y"

    prompt = get_chat_prompt(SYSTEM_ROLE, PROMPT_INSTRUCTION, context, None, query)

    assert "User Resume/Profile:" not in prompt
    assert "Doc B" in prompt
    assert "Query Y" in prompt
