import pandas as pd
from evidently import Report
from evidently.metrics import DataDriftPreset, TextOverviewPreset
from evidently.ui.workspace import Workspace
import os
import json

def create_evidently_dashboard(workspace_path: str, data_path: str, project_name: str):
    """
    Creates an Evidently Workspace and adds a data drift report to it for the UI.
    """
    print(f"Preparing Evidently workspace at: {workspace_path}")
    os.makedirs(workspace_path, exist_ok=True)
    ws = Workspace.create(workspace_path)
    
    print(f"Loading reference data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        return

    reference_df = pd.read_csv(data_path)
    
    # Create reports using both DataDrift and TextOverview presets
    report = Report(metrics=[
        DataDriftPreset(),
        TextOverviewPreset(column_name='Resume_str')
    ])
    
    # Run report with same data as reference and current (for initialization)
    sample_data = reference_df.sample(min(500, len(reference_df)), replace=True)
    report.run(reference_data=reference_df, current_data=sample_data)
    
    # Create and configure the project
    project = ws.create_project(project_name)
    project.description = "Monitor Resume data drift and text metrics"
    
    # Add the report to the project
    ws.add_report(project.id, report)
    
    # Create dashboard configuration
    dashboard_config = {
        "dashboard": {
            "tabs": [{
                "name": "Data Drift",
                "widgets": [
                    {
                        "title": "Data Drift Score",
                        "size": 2,
                        "type": "metric",
                        "metric": "current",
                        "metricName": "DatasetDriftMetric.result.drift_score"
                    },
                    {
                        "title": "Text Length Distribution",
                        "size": 4,
                        "type": "plot",
                        "metric": "current",
                        "metricName": "TextOverviewPreset.result.current.length_distribution"
                    }
                ]
            }]
        }
    }
    
    # Save dashboard configuration
    project_dir = os.path.join(workspace_path, project.id)
    os.makedirs(project_dir, exist_ok=True)
    
    metadata = {
        "id": project.id,
        "name": project_name,
        "description": project.description,
        "dashboard": dashboard_config["dashboard"]
    }
    
    with open(os.path.join(project_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Evidently workspace initialized successfully.")
    print(f"Project '{project_name}' created with ID: {project.id}")

if __name__ == "__main__":
    create_evidently_dashboard(
        workspace_path="/app/workspace",
        data_path="/app/data/raw/Resume.csv",
        project_name="Resume Data Drift"
    )