import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMMetrics:
    """Container for LLM metrics."""
    request_id: str
    timestamp: datetime
    provider: LLMProvider
    model_name: str
    
    # Latency metrics
    total_latency_ms: float
    ttft_ms: Optional[float] = None  # Time to first token
    
    # Token metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost metrics
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    
    # Quality metrics
    guardrail_violations: int = 0
    output_blocked: bool = False
    
    # Request/Response data
    input_length: int = 0
    output_length: int = 0
    
    # Error tracking
    error: Optional[str] = None
    success: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/storage."""
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider.value,
            "model_name": self.model_name,
            "total_latency_ms": self.total_latency_ms,
            "ttft_ms": self.ttft_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
            "guardrail_violations": self.guardrail_violations,
            "output_blocked": self.output_blocked,
            "input_length": self.input_length,
            "output_length": self.output_length,
            "error": self.error,
            "success": self.success,
        }


class TokenCounter:
    """
    Token counting for various LLM providers.
    
    For production, use tiktoken for OpenAI or provider-specific libraries.
    This is a simple approximation.
    """
    
    @staticmethod
    def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Estimate token count.
        Rule of thumb: ~4 characters = 1 token for English text.
        """
        # Simple approximation
        # For production, use: tiktoken.encoding_for_model(model).encode(text)
        return len(text) // 4
    
    @staticmethod
    def count_tokens_exact(text: str, model: str = "gpt-3.5-turbo") -> int:
        """
        Count exact tokens using tiktoken (if available).
        Falls back to estimation if not available.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except ImportError:
            logger.warning("tiktoken not available, using estimation")
            return TokenCounter.estimate_tokens(text, model)
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}, using estimation")
            return TokenCounter.estimate_tokens(text, model)


class CostCalculator:
    """
    Calculate costs for LLM API calls.
    
    Pricing as of Dec 2024 (update regularly).
    """
    
    # Pricing per 1M tokens (USD)
    PRICING = {
        LLMProvider.OPENAI: {
            "gpt-4": {"prompt": 30.0, "completion": 60.0},
            "gpt-4-turbo": {"prompt": 10.0, "completion": 30.0},
            "gpt-3.5-turbo": {"prompt": 0.5, "completion": 1.5},
            "gpt-4o": {"prompt": 5.0, "completion": 15.0},
            "gpt-4o-mini": {"prompt": 0.15, "completion": 0.6},
        },
        LLMProvider.ANTHROPIC: {
            "claude-3-opus": {"prompt": 15.0, "completion": 75.0},
            "claude-3-sonnet": {"prompt": 3.0, "completion": 15.0},
            "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
        },
        LLMProvider.GROQ: {
            "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
            "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
            "mixtral-8x7b": {"prompt": 0.24, "completion": 0.24},
        },
        LLMProvider.LOCAL: {
            "default": {"prompt": 0.0, "completion": 0.0},
        },
    }
    
    @staticmethod
    def calculate_cost(
        provider: LLMProvider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate cost for an LLM API call.
        
        Returns:
            Dict with prompt_cost, completion_cost, and total_cost
        """
        pricing = CostCalculator.PRICING.get(provider, {})
        model_pricing = pricing.get(model, pricing.get("default", {"prompt": 0, "completion": 0}))
        
        # Calculate cost (pricing is per 1M tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * model_pricing["completion"]
        total_cost = prompt_cost + completion_cost
        
        return {
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "total_cost": total_cost,
        }


class LatencyTracker:
    """Track latency for LLM requests."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def start(self):
        """Mark request start."""
        self.start_time = time.time()
    
    def mark_first_token(self):
        """Mark time to first token (for streaming)."""
        if self.start_time:
            self.first_token_time = time.time()
    
    def stop(self):
        """Mark request end."""
        self.end_time = time.time()
    
    def get_total_latency_ms(self) -> float:
        """Get total latency in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
    
    def get_ttft_ms(self) -> Optional[float]:
        """Get time to first token in milliseconds."""
        if self.start_time and self.first_token_time:
            return (self.first_token_time - self.start_time) * 1000
        return None


class LLMMetricsTracker:
    """
    Comprehensive LLM metrics tracker.
    
    Usage:
        tracker = LLMMetricsTracker()
        
        # Start tracking
        tracker.start_request(provider=LLMProvider.OPENAI, model="gpt-4")
        
        # Make LLM call
        response = llm.generate(prompt)
        
        # Record metrics
        tracker.record_tokens(prompt_tokens=100, completion_tokens=50)
        tracker.record_guardrail_check(violations=2, blocked=False)
        tracker.end_request(success=True)
        
        # Get metrics
        metrics = tracker.get_metrics()
    """
    
    def __init__(self):
        self.current_request_id: Optional[str] = None
        self.provider: Optional[LLMProvider] = None
        self.model_name: Optional[str] = None
        self.latency_tracker = LatencyTracker()
        
        # Metrics storage
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.guardrail_violations = 0
        self.output_blocked = False
        self.input_length = 0
        self.output_length = 0
        self.error: Optional[str] = None
        
        # History
        self.metrics_history: List[LLMMetrics] = []
    
    def start_request(
        self,
        provider: LLMProvider,
        model: str,
        request_id: Optional[str] = None
    ):
        """Start tracking a new request."""
        self.current_request_id = request_id or self._generate_request_id()
        self.provider = provider
        self.model_name = model
        self.latency_tracker.start()
        
        # Reset metrics
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.guardrail_violations = 0
        self.output_blocked = False
        self.input_length = 0
        self.output_length = 0
        self.error = None
    
    def record_input(self, input_text: str):
        """Record input text and estimate tokens."""
        self.input_length = len(input_text)
        self.prompt_tokens = TokenCounter.estimate_tokens(input_text, self.model_name)
    
    def record_output(self, output_text: str):
        """Record output text and estimate tokens."""
        self.output_length = len(output_text)
        self.completion_tokens = TokenCounter.estimate_tokens(output_text, self.model_name)
    
    def record_tokens(self, prompt_tokens: int, completion_tokens: int):
        """Record exact token counts from API response."""
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
    
    def record_guardrail_check(self, violations: int, blocked: bool):
        """Record guardrail check results."""
        self.guardrail_violations = violations
        self.output_blocked = blocked
    
    def record_error(self, error: Exception):
        """Record error."""
        self.error = str(error)
    
    def end_request(self, success: bool = True) -> LLMMetrics:
        """
        End request tracking and return metrics.
        
        Args:
            success: Whether the request was successful
            
        Returns:
            LLMMetrics object with all tracked metrics
        """
        self.latency_tracker.stop()
        
        # Calculate costs
        total_tokens = self.prompt_tokens + self.completion_tokens
        costs = CostCalculator.calculate_cost(
            self.provider,
            self.model_name,
            self.prompt_tokens,
            self.completion_tokens
        )
        
        # Create metrics object
        metrics = LLMMetrics(
            request_id=self.current_request_id,
            timestamp=datetime.now(),
            provider=self.provider,
            model_name=self.model_name,
            total_latency_ms=self.latency_tracker.get_total_latency_ms(),
            ttft_ms=self.latency_tracker.get_ttft_ms(),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=total_tokens,
            prompt_cost=costs["prompt_cost"],
            completion_cost=costs["completion_cost"],
            total_cost=costs["total_cost"],
            guardrail_violations=self.guardrail_violations,
            output_blocked=self.output_blocked,
            input_length=self.input_length,
            output_length=self.output_length,
            error=self.error,
            success=success,
        )
        
        # Store in history
        self.metrics_history.append(metrics)
        
        # Log metrics
        logger.info(
            f"LLM Request [{self.current_request_id}]: "
            f"provider={self.provider.value}, model={self.model_name}, "
            f"latency={metrics.total_latency_ms:.2f}ms, "
            f"tokens={total_tokens}, cost=${metrics.total_cost:.6f}"
        )
        
        return metrics
    
    def get_metrics(self) -> Optional[LLMMetrics]:
        """Get metrics for current request."""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_history(self, limit: int = 100) -> List[LLMMetrics]:
        """Get recent metrics history."""
        return self.metrics_history[-limit:]
    
    def get_aggregate_stats(self) -> Dict:
        """Get aggregate statistics across all tracked requests."""
        if not self.metrics_history:
            return {}
        
        total_requests = len(self.metrics_history)
        successful_requests = sum(1 for m in self.metrics_history if m.success)
        
        total_tokens = sum(m.total_tokens for m in self.metrics_history)
        total_cost = sum(m.total_cost for m in self.metrics_history)
        avg_latency = sum(m.total_latency_ms for m in self.metrics_history) / total_requests
        
        total_violations = sum(m.guardrail_violations for m in self.metrics_history)
        blocked_outputs = sum(1 for m in self.metrics_history if m.output_blocked)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_latency_ms": avg_latency,
            "total_guardrail_violations": total_violations,
            "blocked_outputs": blocked_outputs,
        }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
