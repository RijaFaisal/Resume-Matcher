from prometheus_client import Counter, Histogram, Gauge, Summary, Info
from typing import Optional
import logging

from .llm_metrics import LLMMetrics, LLMProvider

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics exporter for LLM monitoring.
    
    Metrics exposed:
    - llm_requests_total: Total LLM API requests
    - llm_request_duration_seconds: Request latency histogram
    - llm_tokens_total: Total tokens processed
    - llm_cost_dollars_total: Total cost in USD
    - llm_guardrail_violations_total: Total guardrail violations
    - llm_errors_total: Total errors
    - llm_ttft_seconds: Time to first token histogram
    """
    
    def __init__(self):
        """Initialize Prometheus metrics."""
        
        # Request counters
        self.requests_total = Counter(
            'llm_requests_total',
            'Total number of LLM API requests',
            ['provider', 'model', 'status']
        )
        
        # Latency metrics
        self.request_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration in seconds',
            ['provider', 'model'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        self.ttft = Histogram(
            'llm_ttft_seconds',
            'Time to first token in seconds',
            ['provider', 'model'],
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
        )
        
        # Token metrics
        self.tokens_total = Counter(
            'llm_tokens_total',
            'Total tokens processed',
            ['provider', 'model', 'type']  # type: prompt or completion
        )
        
        self.tokens_per_request = Histogram(
            'llm_tokens_per_request',
            'Tokens per request distribution',
            ['provider', 'model'],
            buckets=(10, 50, 100, 500, 1000, 2000, 5000, 10000, 50000)
        )
        
        # Cost metrics
        self.cost_dollars_total = Counter(
            'llm_cost_dollars_total',
            'Total cost in USD',
            ['provider', 'model']
        )
        
        self.cost_per_request = Histogram(
            'llm_cost_per_request_dollars',
            'Cost per request in USD',
            ['provider', 'model'],
            buckets=(0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0)
        )
        
        # Guardrail metrics
        self.guardrail_violations_total = Counter(
            'llm_guardrail_violations_total',
            'Total guardrail violations',
            ['provider', 'model', 'severity']  # severity: input or output
        )
        
        self.blocked_outputs_total = Counter(
            'llm_blocked_outputs_total',
            'Total blocked outputs due to guardrails',
            ['provider', 'model']
        )
        
        # Error metrics
        self.errors_total = Counter(
            'llm_errors_total',
            'Total LLM errors',
            ['provider', 'model', 'error_type']
        )
        
        # Current state gauges
        self.current_requests = Gauge(
            'llm_current_requests',
            'Number of LLM requests currently in flight',
            ['provider', 'model']
        )
        
        # Summary statistics
        self.token_usage_summary = Summary(
            'llm_token_usage',
            'Summary of token usage',
            ['provider', 'model']
        )
        
        # Model info
        self.model_info = Info(
            'llm_model',
            'LLM model information'
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_request(self, metrics: LLMMetrics):
        """
        Record all metrics from an LLM request.
        
        Args:
            metrics: LLMMetrics object containing request metrics
        """
        provider = metrics.provider.value
        model = metrics.model_name
        status = "success" if metrics.success else "error"
        
        # Request counter
        self.requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        # Latency
        if metrics.total_latency_ms:
            self.request_duration.labels(
                provider=provider,
                model=model
            ).observe(metrics.total_latency_ms / 1000)  # Convert to seconds
        
        # Time to first token
        if metrics.ttft_ms:
            self.ttft.labels(
                provider=provider,
                model=model
            ).observe(metrics.ttft_ms / 1000)
        
        # Token metrics
        if metrics.prompt_tokens:
            self.tokens_total.labels(
                provider=provider,
                model=model,
                type="prompt"
            ).inc(metrics.prompt_tokens)
        
        if metrics.completion_tokens:
            self.tokens_total.labels(
                provider=provider,
                model=model,
                type="completion"
            ).inc(metrics.completion_tokens)
        
        if metrics.total_tokens:
            self.tokens_per_request.labels(
                provider=provider,
                model=model
            ).observe(metrics.total_tokens)
            
            self.token_usage_summary.labels(
                provider=provider,
                model=model
            ).observe(metrics.total_tokens)
        
        # Cost metrics
        if metrics.total_cost:
            self.cost_dollars_total.labels(
                provider=provider,
                model=model
            ).inc(metrics.total_cost)
            
            self.cost_per_request.labels(
                provider=provider,
                model=model
            ).observe(metrics.total_cost)
        
        # Guardrail metrics
        if metrics.guardrail_violations:
            self.guardrail_violations_total.labels(
                provider=provider,
                model=model,
                severity="total"
            ).inc(metrics.guardrail_violations)
        
        if metrics.output_blocked:
            self.blocked_outputs_total.labels(
                provider=provider,
                model=model
            ).inc()
        
        # Error metrics
        if not metrics.success and metrics.error:
            error_type = type(metrics.error).__name__ if isinstance(metrics.error, Exception) else "unknown"
            self.errors_total.labels(
                provider=provider,
                model=model,
                error_type=error_type
            ).inc()
    
    def increment_current_requests(self, provider: str, model: str):
        """Increment in-flight requests counter."""
        self.current_requests.labels(provider=provider, model=model).inc()
    
    def decrement_current_requests(self, provider: str, model: str):
        """Decrement in-flight requests counter."""
        self.current_requests.labels(provider=provider, model=model).dec()
    
    def set_model_info(self, provider: str, model: str, version: str = "unknown"):
        """Set model information."""
        self.model_info.info({
            'provider': provider,
            'model': model,
            'version': version
        })


# Global Prometheus metrics instance
prometheus_metrics = PrometheusMetrics()


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics instance."""
    return prometheus_metrics
