import logging
import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .input_validators import InputValidator, ValidationResult
from .output_moderators import OutputModerator, ModerationResult, ModerationAction

logger = logging.getLogger(__name__)


class PolicyMode(Enum):
    """Policy enforcement modes."""
    PERMISSIVE = "permissive"  # Log violations but allow
    BALANCED = "balanced"      # Block critical violations
    STRICT = "strict"          # Block all violations


@dataclass
class GuardrailsConfig:
    """Configuration for guardrails system."""
    
    # Policy mode
    mode: PolicyMode = PolicyMode.BALANCED
    
    # Input validation settings
    enable_pii_detection: bool = True
    enable_injection_filter: bool = True
    mask_pii: bool = True
    strict_injection_filter: bool = False
    max_input_length: int = 50000
    min_input_length: int = 10
    
    # Output moderation settings
    enable_toxicity_filter: bool = True
    enable_hallucination_detector: bool = True
    toxicity_threshold: float = 0.7
    confidence_threshold: float = 0.6
    
    # Logging and monitoring
    log_violations: bool = True
    log_all_checks: bool = False
    
    # Rate limiting
    enable_rate_limiting: bool = False
    max_requests_per_minute: int = 60
    
    # Metrics
    collect_metrics: bool = True
    
    # Fallback behavior
    block_on_error: bool = True  # Block if guardrails fail


@dataclass
class GuardrailMetrics:
    """Metrics for guardrails monitoring."""
    total_requests: int = 0
    input_violations: int = 0
    output_violations: int = 0
    blocked_requests: int = 0
    total_latency_ms: float = 0.0
    
    def add_latency(self, latency_ms: float):
        """Add a latency measurement."""
        self.total_latency_ms += latency_ms
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "input_violations": self.input_violations,
            "output_violations": self.output_violations,
            "blocked_requests": self.blocked_requests,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "violation_rate": round(
                (self.input_violations + self.output_violations) / max(self.total_requests, 1),
                4
            ),
        }


@dataclass
class GuardrailResult:
    """Complete guardrail check result."""
    allowed: bool
    input_validation: Optional[ValidationResult] = None
    output_moderation: Optional[ModerationResult] = None
    sanitized_input: Optional[str] = None
    filtered_output: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyEngine:
    """
    Central policy engine that orchestrates all guardrail components.
    
    Usage:
        engine = PolicyEngine(config)
        
        # Validate input
        result = engine.validate_input(user_input)
        if not result.allowed:
            return error_response(result)
        
        # Generate response
        llm_output = generate_response(result.sanitized_input or user_input)
        
        # Moderate output
        result = engine.moderate_output(llm_output, context=user_input)
        if not result.allowed:
            return filtered_response(result)
        
        return llm_output
    """

    def __init__(self, config: Optional[GuardrailsConfig] = None):
        """
        Initialize policy engine with configuration.
        
        Args:
            config: GuardrailsConfig instance. Uses defaults if None.
        """
        self.config = config or GuardrailsConfig()
        self.metrics = GuardrailMetrics()
        
        # Initialize validators
        self.input_validator = InputValidator(
            enable_pii_detection=self.config.enable_pii_detection,
            enable_injection_filter=self.config.enable_injection_filter,
            mask_pii=self.config.mask_pii,
            strict_mode=self.config.strict_injection_filter,
            max_length=self.config.max_input_length,
            min_length=self.config.min_input_length,
        )
        
        # Initialize moderators
        self.output_moderator = OutputModerator(
            enable_toxicity_filter=self.config.enable_toxicity_filter,
            enable_hallucination_detector=self.config.enable_hallucination_detector,
            toxicity_threshold=self.config.toxicity_threshold,
            confidence_threshold=self.config.confidence_threshold,
        )
        
        logger.info(f"PolicyEngine initialized with mode={self.config.mode.value}")

    def validate_input(self, user_input: str) -> GuardrailResult:
        """
        Validate user input against all input guardrails.
        
        Args:
            user_input: User-provided input text
            
        Returns:
            GuardrailResult with validation outcome
        """
        start_time = time.time()
        
        try:
            # Run validation
            validation_result = self.input_validator.validate(user_input)
            
            # Determine if request is allowed based on policy mode
            allowed = self._should_allow_input(validation_result)
            
            # Update metrics
            if self.config.collect_metrics:
                self.metrics.total_requests += 1
                if validation_result.violations:
                    self.metrics.input_violations += 1
                if not allowed:
                    self.metrics.blocked_requests += 1
                self.metrics.add_latency((time.time() - start_time) * 1000)
            
            # Log if enabled
            if self.config.log_violations and validation_result.violations:
                logger.warning(
                    f"Input validation violations: {validation_result.violations}, "
                    f"risk={validation_result.risk_level.value}, allowed={allowed}"
                )
            elif self.config.log_all_checks:
                logger.debug(f"Input validation passed: risk={validation_result.risk_level.value}")
            
            return GuardrailResult(
                allowed=allowed,
                input_validation=validation_result,
                sanitized_input=validation_result.sanitized_input,
                metadata={
                    "policy_mode": self.config.mode.value,
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            )
            
        except Exception as e:
            logger.error(f"Error in input validation: {e}", exc_info=True)
            
            # Fail-safe behavior
            if self.config.block_on_error:
                return GuardrailResult(
                    allowed=False,
                    metadata={"error": str(e), "blocked_on_error": True}
                )
            else:
                return GuardrailResult(
                    allowed=True,
                    metadata={"error": str(e), "allowed_despite_error": True}
                )

    def moderate_output(
        self, 
        llm_output: str, 
        context: Optional[str] = None
    ) -> GuardrailResult:
        """
        Moderate LLM output against all output guardrails.
        
        Args:
            llm_output: Generated output from LLM
            context: Original context/input for grounding check
            
        Returns:
            GuardrailResult with moderation outcome
        """
        start_time = time.time()
        
        try:
            # Run moderation
            moderation_result = self.output_moderator.moderate(llm_output, context)
            
            # Determine if output is allowed based on policy mode
            allowed = self._should_allow_output(moderation_result)
            
            # Update metrics
            if self.config.collect_metrics:
                if moderation_result.violations:
                    self.metrics.output_violations += 1
                if not allowed:
                    self.metrics.blocked_requests += 1
                self.metrics.add_latency((time.time() - start_time) * 1000)
            
            # Log if enabled
            if self.config.log_violations and moderation_result.violations:
                logger.warning(
                    f"Output moderation violations: {moderation_result.violations}, "
                    f"action={moderation_result.action.value}, allowed={allowed}"
                )
            elif self.config.log_all_checks:
                logger.debug(f"Output moderation passed: action={moderation_result.action.value}")
            
            return GuardrailResult(
                allowed=allowed,
                output_moderation=moderation_result,
                filtered_output=moderation_result.filtered_output,
                metadata={
                    "policy_mode": self.config.mode.value,
                    "latency_ms": (time.time() - start_time) * 1000,
                }
            )
            
        except Exception as e:
            logger.error(f"Error in output moderation: {e}", exc_info=True)
            
            # Fail-safe behavior
            if self.config.block_on_error:
                return GuardrailResult(
                    allowed=False,
                    filtered_output="[Error: Output could not be verified for safety]",
                    metadata={"error": str(e), "blocked_on_error": True}
                )
            else:
                return GuardrailResult(
                    allowed=True,
                    metadata={"error": str(e), "allowed_despite_error": True}
                )

    def _should_allow_input(self, validation: ValidationResult) -> bool:
        """Determine if input should be allowed based on policy mode."""
        if self.config.mode == PolicyMode.PERMISSIVE:
            return True
        elif self.config.mode == PolicyMode.BALANCED:
            # Allow LOW risk, block MEDIUM and above
            return validation.is_valid
        else:  # STRICT
            # Block anything with violations
            return validation.is_valid and not validation.violations

    def _should_allow_output(self, moderation: ModerationResult) -> bool:
        """Determine if output should be allowed based on policy mode."""
        if self.config.mode == PolicyMode.PERMISSIVE:
            return moderation.action != ModerationAction.BLOCK
        elif self.config.mode == PolicyMode.BALANCED:
            return moderation.action in [ModerationAction.ALLOW, ModerationAction.WARN]
        else:  # STRICT
            return moderation.action == ModerationAction.ALLOW

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """Reset metrics counters."""
        self.metrics = GuardrailMetrics()
        logger.info("Guardrails metrics reset")
