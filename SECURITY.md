# 🛡️ Security & AI Safety Policy

## 🔒 Data Privacy & PII Protection

We take data privacy seriously. The Resume Matcher deals with sensitive personal resume data, and we have implemented strict measures to protect it.

### 1. Local-First Processing
*   **Vector Storage**: All resume embeddings are stored in a local **FAISS** index (`vectorstore/`). No user data is sent to external vector databases.
*   **Ephemeral processing**: Resume text is processed in-memory for the duration of the request and is not persistently logged in its raw form.

### 2. Automatic PII Redaction
We use a regex-based `PIIDetector` (located in `src/guardrails/input_validators.py`) to automatically detect and redact sensitive fields *before* they are processed or logged.

**Redacted Fields:**
*   📧 Email Addresses (`[EMAIL_REDACTED]`)
*   📱 Phone Numbers (`[PHONE_US_REDACTED]`)
*   🏠 Physical Addresses
*   💳 Credit Card Numbers & SSNs

---

## 🛑 Guardrails & Defense Mechanisms

We have implemented a multi-layer guardrails system to ensure the AI behaves responsibly and securely.

### 1. Input Validation (`InputValidator`)
Before any text reaches the LLM, it passes through strict filters:

*   **Prompt Injection Defense**: 
    *   We use a `PromptInjectionFilter` that scans for adversarial patterns like *"Ignore previous instructions"*, *"System override"*, or delimiter confusion attacks.
    *   **Risk Levels**: Attacks classified as `CRITICAL` (e.g., SQL Injection patterns) are immediately blocked.
*   **Toxicity Filter**: 
    *   Blocks input containing hate speech, violence, or sexual content to prevent the model from being goaded into toxic conversations.

### 2. Output Moderation (`OutputModerator`)
We don't trust the model blindly. All generated responses are validated:

*   **Hallucination Detection**:
    *   We scan responses for uncertainty markers (*"I'm not sure", "maybe"*) and conflicting statements.
    *   **Grounding Check**: Verifies that the skills mentioned in the advice actually exist in the provided job/resume context.
*   **Toxicity Check**: 
    *   Ensures the career advice remains professional and free of offensive language.

---

## 🤖 Responsible AI Guidelines

The "Career Coach" persona is strictly scoped:

1.  **Professional Scope**: The AI is instructed to ONLY provide career and technical advice. It will refuse to answer questions about politics, religion, or medical issues.
2.  **No Guarantee**: The model output is for advisory purposes only and does not guarantee job placement.
3.  **Transparency**: Users are informed that they are interacting with an AI system.

---

## 🐛 Reporting Vulnerabilities

If you discover a security vulnerability or a prompt injection bypass, please do not disclose it publicly.
**Contact**: security@resume-matcher-project.com
