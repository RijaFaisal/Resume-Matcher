from src.guardrails import GuardrailsConfig, PolicyMode


# ============================================================================
# DEVELOPMENT CONFIGURATION
# ============================================================================
DEV_CONFIG = GuardrailsConfig(
    mode=PolicyMode.PERMISSIVE,  # Log violations but don't block
    enable_pii_detection=True,
    enable_injection_filter=True,
    enable_toxicity_filter=True,
    enable_hallucination_detector=True,
    mask_pii=False,  # Don't mask during development
    strict_injection_filter=False,
    max_input_length=100000,
    min_input_length=5,
    toxicity_threshold=0.9,  # Very lenient
    confidence_threshold=0.3,  # Very lenient
    log_violations=True,
    log_all_checks=True,  # Verbose logging
    collect_metrics=True,
    block_on_error=False,  # Don't block on guardrail errors
)


# ============================================================================
# PRODUCTION CONFIGURATION (RECOMMENDED)
# ============================================================================
PRODUCTION_CONFIG = GuardrailsConfig(
    mode=PolicyMode.BALANCED,  # Block medium+ violations
    enable_pii_detection=True,
    enable_injection_filter=True,
    enable_toxicity_filter=True,
    enable_hallucination_detector=True,
    mask_pii=True,  # Always mask PII in production
    strict_injection_filter=False,
    max_input_length=50000,
    min_input_length=10,
    toxicity_threshold=0.7,
    confidence_threshold=0.6,
    log_violations=True,
    log_all_checks=False,  # Only log violations
    collect_metrics=True,
    block_on_error=True,  # Fail safe
)


# ============================================================================
# HIGH SECURITY CONFIGURATION
# ============================================================================
HIGH_SECURITY_CONFIG = GuardrailsConfig(
    mode=PolicyMode.STRICT,  # Block any violations
    enable_pii_detection=True,
    enable_injection_filter=True,
    enable_toxicity_filter=True,
    enable_hallucination_detector=True,
    mask_pii=True,
    strict_injection_filter=True,  # More aggressive filtering
    max_input_length=25000,  # Shorter inputs
    min_input_length=10,
    toxicity_threshold=0.5,  # Very strict
    confidence_threshold=0.8,  # High confidence required
    log_violations=True,
    log_all_checks=True,
    collect_metrics=True,
    block_on_error=True,
)


# ============================================================================
# RESUME PROCESSING CONFIGURATION
# ============================================================================
RESUME_CONFIG = GuardrailsConfig(
    mode=PolicyMode.BALANCED,
    enable_pii_detection=True,  # Important for resumes
    enable_injection_filter=True,
    enable_toxicity_filter=False,  # Resumes may contain industry terms
    enable_hallucination_detector=False,  # Not applicable
    mask_pii=True,
    strict_injection_filter=False,
    max_input_length=100000,  # Resumes can be long
    min_input_length=50,
    toxicity_threshold=0.8,
    confidence_threshold=0.5,
    log_violations=True,
    log_all_checks=False,
    collect_metrics=True,
    block_on_error=True,
)


# ============================================================================
# RAG/CHATBOT CONFIGURATION
# ============================================================================
RAG_CONFIG = GuardrailsConfig(
    mode=PolicyMode.BALANCED,
    enable_pii_detection=True,
    enable_injection_filter=True,
    enable_toxicity_filter=True,  # Important for chatbots
    enable_hallucination_detector=True,  # Critical for RAG
    mask_pii=True,
    strict_injection_filter=True,  # Chatbots are high-risk
    max_input_length=10000,
    min_input_length=10,
    toxicity_threshold=0.7,
    confidence_threshold=0.6,
    log_violations=True,
    log_all_checks=False,
    collect_metrics=True,
    block_on_error=True,
)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
from src.guardrails import PolicyEngine
from src.guardrails.config.example import PRODUCTION_CONFIG

# Initialize with production config
engine = PolicyEngine(PRODUCTION_CONFIG)

# Validate input
result = engine.validate_input(user_input)
if not result.allowed:
    return {"error": "Invalid input", "violations": result.input_validation.violations}

# Moderate output
result = engine.moderate_output(llm_output, context=user_input)
if not result.allowed:
    return {"output": result.filtered_output}

return {"output": llm_output}
"""
