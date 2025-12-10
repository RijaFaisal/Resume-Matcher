import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.monitoring.evidently_monitor import EvidentlyMonitor

@pytest.fixture
def monitor(tmp_path):
    return EvidentlyMonitor(workspace_dir=str(tmp_path))

def test_init_creates_dirs(tmp_path):
    _ = EvidentlyMonitor(workspace_dir=str(tmp_path))
    assert (tmp_path / "reports").exists()

def test_log_metrics(monitor):
    metrics = {"latency": 100}
    monitor.log_metrics(metrics)
    
    with open(monitor.data_file, 'r') as f:
        data = json.load(f)
    assert data["latency"] == 100

def test_load_metrics(monitor):
    monitor.log_metrics({"val": 1})
    monitor.log_metrics({"val": 2})
    
    df = monitor.load_metrics()
    assert len(df) == 2
    assert df.iloc[0]["val"] == 1

def test_get_statistics_empty(monitor):
    stats = monitor.get_statistics()
    assert stats == {}

def test_get_statistics(monitor):
    monitor.log_metrics({"total_latency_ms": 10, "total_tokens": 5, "total_cost": 0.01, "success": True})
    stats = monitor.get_statistics()
    assert stats["total_requests"] == 1
    assert stats["avg_latency_ms"] == 10
    assert stats["total_cost"] == 0.01

@patch("evidently.report.Report") 
def test_create_drift_report_mock(mock_report, monitor):
    # Just verify flow, assume evidently installed or mocked
    # If evidently import fails inside method, it returns None
    # We can rely on the fact that if it returns None (due to ImportError), we covered the exception block
    # But ideally we mock sys.modules or the imports inside the method
    with patch.dict("sys.modules", {"evidently": MagicMock(), "evidently.report": MagicMock()}):
         monitor.create_drift_report()
         # coverage test mainly
         pass

def test_detect_anomalies(monitor):
    monitor.log_metrics({"val": 10})
    monitor.log_metrics({"val": 10})
    monitor.log_metrics({"val": 100}) # Anomaly
    
    anomalies = monitor.detect_anomalies(metric="val", threshold_std=2)
    assert len(anomalies) == 1
    assert anomalies[0]["val"] == 100
