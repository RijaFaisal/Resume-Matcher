import pandas as pd
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EvidentlyMonitor:
    """
    Evidently AI monitoring for LLM data drift and quality.
    
    Monitors:
    - Input distribution drift
    - Output quality degradation
    - Token usage patterns
    - Cost trends
    - Guardrail violation trends
    """
    
    def __init__(self, workspace_dir: str = "./monitoring/evidently/workspace"):
        """
        Initialize Evidently monitor.
        
        Args:
            workspace_dir: Directory for Evidently workspace and reports
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_file = self.workspace_dir / "llm_metrics.jsonl"
        self.reports_dir = self.workspace_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info(f"Evidently monitor initialized at {self.workspace_dir}")
    
    def log_metrics(self, metrics_dict: Dict):
        """
        Log metrics to JSONL file for Evidently processing.
        
        Args:
            metrics_dict: Dictionary containing metrics
        """
        with open(self.data_file, 'a') as f:
            json.dump(metrics_dict, f)
            f.write('\n')
    
    def load_metrics(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load metrics from JSONL file.
        
        Args:
            limit: Maximum number of records to load
            
        Returns:
            DataFrame with metrics
        """
        if not self.data_file.exists():
            return pd.DataFrame()
        
        data = []
        with open(self.data_file, 'r') as f:
            for line in f:
                if limit and len(data) >= limit:
                    break
                data.append(json.loads(line))
        
        return pd.DataFrame(data)
    
    def create_drift_report(
        self,
        reference_window: int = 1000,
        current_window: int = 100
    ) -> Optional[str]:
        """
        Create data drift report comparing reference and current data.
        
        Args:
            reference_window: Number of samples for reference dataset
            current_window: Number of samples for current dataset
            
        Returns:
            Path to generated HTML report
        """
        try:
            from evidently import ColumnMapping
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset, DataQualityPreset
            
            df = self.load_metrics()
            
            if len(df) < reference_window + current_window:
                logger.warning(
                    f"Not enough data for drift report: "
                    f"{len(df)} < {reference_window + current_window}"
                )
                return None
            
            # Split into reference and current
            reference_data = df.iloc[:reference_window]
            current_data = df.iloc[-current_window:]
            
            # Create column mapping
            column_mapping = ColumnMapping(
                numerical_features=[
                    'total_latency_ms',
                    'prompt_tokens',
                    'completion_tokens',
                    'total_tokens',
                    'total_cost',
                    'input_length',
                    'output_length',
                ],
                categorical_features=[
                    'provider',
                    'model_name',
                    'success',
                ]
            )
            
            # Create report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset(),
            ])
            
            report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=column_mapping
            )
            
            # Save report
            report_path = self.reports_dir / f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            logger.info(f"Drift report saved to {report_path}")
            return str(report_path)
            
        except ImportError:
            logger.error("Evidently not installed. Install with: pip install evidently")
            return None
        except Exception as e:
            logger.error(f"Error creating drift report: {e}")
            return None
    
    def create_performance_report(
        self,
        window: int = 1000
    ) -> Optional[str]:
        """
        Create performance monitoring report.
        
        Args:
            window: Number of recent samples to analyze
            
        Returns:
            Path to generated HTML report
        """
        try:
            from evidently.report import Report
            from evidently.metrics import (
                ColumnSummaryMetric,
                ColumnDistributionMetric,
                DatasetMissingValuesMetric,
            )
            
            df = self.load_metrics(limit=window)
            
            if df.empty:
                logger.warning("No data available for performance report")
                return None
            
            # Create report
            report = Report(metrics=[
                DatasetMissingValuesMetric(),
                ColumnSummaryMetric(column_name='total_latency_ms'),
                ColumnSummaryMetric(column_name='total_tokens'),
                ColumnSummaryMetric(column_name='total_cost'),
                ColumnDistributionMetric(column_name='total_latency_ms'),
                ColumnDistributionMetric(column_name='total_tokens'),
            ])
            
            report.run(current_data=df, reference_data=None)
            
            # Save report
            report_path = self.reports_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(str(report_path))
            
            logger.info(f"Performance report saved to {report_path}")
            return str(report_path)
            
        except ImportError:
            logger.error("Evidently not installed. Install with: pip install evidently")
            return None
        except Exception as e:
            logger.error(f"Error creating performance report: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get summary statistics from collected metrics.
        
        Returns:
            Dictionary with statistics
        """
        df = self.load_metrics()
        
        if df.empty:
            return {}
        
        return {
            "total_requests": len(df),
            "avg_latency_ms": df['total_latency_ms'].mean() if 'total_latency_ms' in df else 0,
            "p95_latency_ms": df['total_latency_ms'].quantile(0.95) if 'total_latency_ms' in df else 0,
            "p99_latency_ms": df['total_latency_ms'].quantile(0.99) if 'total_latency_ms' in df else 0,
            "avg_tokens": df['total_tokens'].mean() if 'total_tokens' in df else 0,
            "total_cost": df['total_cost'].sum() if 'total_cost' in df else 0,
            "success_rate": df['success'].mean() if 'success' in df else 0,
            "models_used": df['model_name'].nunique() if 'model_name' in df else 0,
        }
    
    def analyze_cost_trends(self, window: int = 24) -> Dict:
        """
        Analyze cost trends over time.
        
        Args:
            window: Number of hours to analyze
            
        Returns:
            Dictionary with cost trend analysis
        """
        df = self.load_metrics()
        
        if df.empty or 'timestamp' not in df or 'total_cost' not in df:
            return {}
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to window
        cutoff = datetime.now() - pd.Timedelta(hours=window)
        df = df[df['timestamp'] >= cutoff]
        
        if df.empty:
            return {}
        
        # Group by hour
        df['hour'] = df['timestamp'].dt.floor('H')
        hourly_cost = df.groupby('hour')['total_cost'].sum()
        
        return {
            "window_hours": window,
            "total_cost": df['total_cost'].sum(),
            "avg_cost_per_hour": hourly_cost.mean(),
            "max_cost_per_hour": hourly_cost.max(),
            "cost_trend": "increasing" if hourly_cost.is_monotonic_increasing else "stable",
        }
    
    def detect_anomalies(self, metric: str = 'total_latency_ms', threshold_std: float = 3.0) -> List[Dict]:
        """
        Detect anomalies in metrics using standard deviation method.
        
        Args:
            metric: Metric column to analyze
            threshold_std: Number of standard deviations for anomaly threshold
            
        Returns:
            List of anomalies
        """
        df = self.load_metrics()
        
        if df.empty or metric not in df:
            return []
        
        mean = df[metric].mean()
        std = df[metric].std()
        threshold = threshold_std * std
        
        anomalies = df[abs(df[metric] - mean) > threshold]
        
        return anomalies.to_dict('records')


# Global Evidently monitor instance
evidently_monitor = EvidentlyMonitor()


def get_evidently_monitor() -> EvidentlyMonitor:
    """Get global Evidently monitor instance."""
    return evidently_monitor
