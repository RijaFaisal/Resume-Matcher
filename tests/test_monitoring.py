import pytest
from datetime import datetime
from src.monitoring import (
    LLMMetricsTracker,
    TokenCounter,
    CostCalculator,
    LatencyTracker,
    LLMProvider,
    LLMMetrics,
)


class TestTokenCounter:
    """Test token counting functionality."""
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world, this is a test"
        tokens = TokenCounter.estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count
    
    def test_empty_text(self):
        """Test with empty text."""
        tokens = TokenCounter.estimate_tokens("")
        assert tokens == 0
    
    def test_long_text(self):
        """Test with long text."""
        text = "word " * 1000
        tokens = TokenCounter.estimate_tokens(text)
        assert tokens > 0


class TestCostCalculator:
    """Test cost calculation functionality."""
    
    def test_openai_cost(self):
        """Test OpenAI cost calculation."""
        cost = CostCalculator.calculate_cost(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        assert "total_cost" in cost
        assert "prompt_cost" in cost
        assert "completion_cost" in cost
        assert cost["total_cost"] > 0
        assert cost["total_cost"] == cost["prompt_cost"] + cost["completion_cost"]
    
    def test_groq_cost(self):
        """Test Groq cost calculation."""
        cost = CostCalculator.calculate_cost(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        assert cost["total_cost"] > 0
    
    def test_local_cost(self):
        """Test local model (free) cost."""
        cost = CostCalculator.calculate_cost(
            provider=LLMProvider.LOCAL,
            model="default",
            prompt_tokens=1000,
            completion_tokens=500
        )
        
        assert cost["total_cost"] == 0.0
    
    def test_zero_tokens(self):
        """Test with zero tokens."""
        cost = CostCalculator.calculate_cost(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            prompt_tokens=0,
            completion_tokens=0
        )
        
        assert cost["total_cost"] == 0.0


class TestLatencyTracker:
    """Test latency tracking."""
    
    def test_latency_tracking(self):
        """Test basic latency tracking."""
        tracker = LatencyTracker()
        
        tracker.start()
        import time
        time.sleep(0.1)
        tracker.stop()
        
        latency = tracker.get_total_latency_ms()
        assert latency >= 100  # Should be at least 100ms
        assert latency < 200   # But not much more
    
    def test_ttft_tracking(self):
        """Test time to first token tracking."""
        tracker = LatencyTracker()
        
        tracker.start()
        import time
        time.sleep(0.05)
        tracker.mark_first_token()
        time.sleep(0.05)
        tracker.stop()
        
        ttft = tracker.get_ttft_ms()
        assert ttft is not None
        assert ttft >= 50
        assert ttft < 100
    
    def test_no_stop(self):
        """Test when stop is not called."""
        tracker = LatencyTracker()
        tracker.start()
        
        latency = tracker.get_total_latency_ms()
        assert latency == 0.0


class TestLLMMetricsTracker:
    """Test comprehensive metrics tracking."""
    
    def test_full_tracking_flow(self):
        """Test complete tracking flow."""
        tracker = LLMMetricsTracker()
        
        # Start request
        tracker.start_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        
        # Record input
        input_text = "What is machine learning?"
        tracker.record_input(input_text)
        
        # Simulate processing
        import time
        time.sleep(0.01)
        
        # Record output
        output_text = "Machine learning is a subset of AI..."
        tracker.record_output(output_text)
        
        # Record exact tokens (simulated)
        tracker.record_tokens(prompt_tokens=10, completion_tokens=20)
        
        # Record guardrail check
        tracker.record_guardrail_check(violations=0, blocked=False)
        
        # End request
        metrics = tracker.end_request(success=True)
        
        # Verify metrics
        assert metrics.success is True
        assert metrics.provider == LLMProvider.OPENAI
        assert metrics.model_name == "gpt-4"
        assert metrics.prompt_tokens == 10
        assert metrics.completion_tokens == 20
        assert metrics.total_tokens == 30
        assert metrics.total_latency_ms > 0
        assert metrics.total_cost > 0
        assert metrics.guardrail_violations == 0
        assert metrics.output_blocked is False
    
    def test_error_tracking(self):
        """Test error tracking."""
        tracker = LLMMetricsTracker()
        
        tracker.start_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        
        error = Exception("API timeout")
        tracker.record_error(error)
        
        metrics = tracker.end_request(success=False)
        
        assert metrics.success is False
        assert metrics.error is not None
    
    def test_guardrail_violations(self):
        """Test guardrail violation tracking."""
        tracker = LLMMetricsTracker()
        
        tracker.start_request(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile"
        )
        
        tracker.record_input("Test input")
        tracker.record_output("Test output")
        tracker.record_guardrail_check(violations=5, blocked=True)
        
        metrics = tracker.end_request(success=False)
        
        assert metrics.guardrail_violations == 5
        assert metrics.output_blocked is True
    
    def test_metrics_history(self):
        """Test metrics history tracking."""
        tracker = LLMMetricsTracker()
        
        # Make 3 requests
        for i in range(3):
            tracker.start_request(
                provider=LLMProvider.OPENAI,
                model="gpt-4"
            )
            tracker.record_input(f"Query {i}")
            tracker.record_output(f"Answer {i}")
            tracker.end_request(success=True)
        
        history = tracker.get_history()
        assert len(history) == 3
        
        # Get aggregate stats
        stats = tracker.get_aggregate_stats()
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 3
        assert stats["success_rate"] == 1.0
    
    def test_cost_aggregation(self):
        """Test cost aggregation."""
        tracker = LLMMetricsTracker()
        
        # Make multiple requests
        for _ in range(5):
            tracker.start_request(
                provider=LLMProvider.OPENAI,
                model="gpt-4"
            )
            tracker.record_tokens(prompt_tokens=100, completion_tokens=50)
            tracker.end_request(success=True)
        
        stats = tracker.get_aggregate_stats()
        assert stats["total_tokens"] == 750  # 150 * 5
        assert stats["total_cost"] > 0


class TestLLMMetrics:
    """Test LLMMetrics dataclass."""
    
    def test_metrics_to_dict(self):
        """Test metrics serialization to dict."""
        metrics = LLMMetrics(
            request_id="test-123",
            timestamp=datetime.now(),
            provider=LLMProvider.OPENAI,
            model_name="gpt-4",
            total_latency_ms=100.5,
            prompt_tokens=50,
            completion_tokens=30,
            total_tokens=80,
            total_cost=0.01,
        )
        
        data = metrics.to_dict()
        
        assert data["request_id"] == "test-123"
        assert data["provider"] == "openai"
        assert data["model_name"] == "gpt-4"
        assert data["total_latency_ms"] == 100.5
        assert data["total_tokens"] == 80


class TestPrometheusIntegration:
    """Test Prometheus metrics integration."""
    
    def test_prometheus_import(self):
        """Test that Prometheus metrics can be imported."""
        from src.monitoring import get_prometheus_metrics
        
        metrics = get_prometheus_metrics()
        assert metrics is not None


class TestEvidentlyIntegration:
    """Test Evidently monitoring integration."""
    
    def test_evidently_import(self):
        """Test that Evidently monitor can be imported."""
        from src.monitoring import get_evidently_monitor
        
        monitor = get_evidently_monitor()
        assert monitor is not None
    
    def test_log_metrics(self):
        """Test logging metrics to Evidently."""
        from src.monitoring import get_evidently_monitor
        
        monitor = get_evidently_monitor()
        
        test_metrics = {
            "request_id": "test-123",
            "provider": "openai",
            "model_name": "gpt-4",
            "total_tokens": 100,
            "total_cost": 0.01,
        }
        
        # Should not raise exception
        monitor.log_metrics(test_metrics)
    
    def test_get_statistics(self):
        """Test getting statistics from Evidently."""
        from src.monitoring import get_evidently_monitor
        
        monitor = get_evidently_monitor()
        stats = monitor.get_statistics()
        
        # Should return dict (may be empty)
        assert isinstance(stats, dict)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_complete_request_flow(self):
        """Test complete request flow with all monitoring."""
        from src.monitoring import (
            LLMMetricsTracker,
            get_prometheus_metrics,
            get_evidently_monitor,
            LLMProvider
        )
        
        # Initialize
        tracker = LLMMetricsTracker()
        prometheus = get_prometheus_metrics()
        evidently = get_evidently_monitor()
        
        # Track request
        tracker.start_request(
            provider=LLMProvider.OPENAI,
            model="gpt-4"
        )
        
        tracker.record_input("What is AI?")
        tracker.record_output("AI is artificial intelligence...")
        tracker.record_tokens(prompt_tokens=10, completion_tokens=20)
        tracker.record_guardrail_check(violations=0, blocked=False)
        
        metrics = tracker.end_request(success=True)
        
        # Record to Prometheus (should not raise)
        prometheus.record_request(metrics)
        
        # Log to Evidently (should not raise)
        evidently.log_metrics(metrics.to_dict())
        
        # Verify
        assert metrics.success is True
        assert metrics.total_cost > 0
