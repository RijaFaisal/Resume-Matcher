from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

TOKENS_PROCESSED_COUNTER = Counter(
    "tokens_processed_total",
    "Total number of tokens processed by the model."
)

def get_instrumentator():
    """Returns a configured Prometheus Instrumentator."""
    return Instrumentator(excluded_handlers=["/health", "/metrics"])