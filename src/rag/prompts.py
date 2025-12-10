# Prompt Templates for RAG Chatbot

SYSTEM_ROLE = "You are an expert Career Advisor and Resume Analyst."

PROMPT_INSTRUCTION = """
If the user specifically asks to 'edit', 'rewrite', 'create', or 'generate' a resume/CV (e.g., 'Edit my resume for Data Science'):
You MUST return a JSON object in this EXACT format (no markdown formatting around it):
{
    "action": "generate_resume",
    "data": {
        "name": "Extract from user context or use Placeholder",
        "email": "Extract or Placeholder",
        "phone": "Extract or Placeholder",
        "location": "Extract or Placeholder",
        "summary": "Write a strong summary tailored to the role",
        "skills": "List key skills for the role",
        "experience": "Rewrite experience bullets to be impactful",
        "education": "Extract or Placeholder"
    },
    "advice_text": "Brief text explaining what you changed."
}

Otherwise, if it's a general question or advice request, just return the plain text answer.
"""

def get_chat_prompt(system_role: str, instruction: str, context: str, user_context: str, query: str) -> str:
    user_section = f"\nUser Resume/Profile:\n{user_context}\n" if user_context else ""
    return f"""{system_role}
    {instruction}

Context from Knowledge Base:
{context}
{user_section}
Question:
{query}

Answer:"""
