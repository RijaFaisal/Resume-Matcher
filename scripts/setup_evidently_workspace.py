import os

import pandas as pd
from evidently.metric_preset import DataDriftPreset, TextOverviewPreset
from evidently.report import Report
from evidently.ui.workspace import Workspace


def main():
    workspace_path = os.path.join("monitoring", "evidently", "workspace")
    data_path = os.path.join("data", "raw", "Resume.csv")

    os.makedirs(workspace_path, exist_ok=True)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    reference_df = df.sample(frac=0.8, random_state=42)
    current_df = df.drop(reference_df.index)
    if len(current_df) == 0:
        current_df = df.sample(min(1000, len(df)), random_state=43)

    report = Report([TextOverviewPreset(column_name="Resume_str"), DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    ws = Workspace.create(workspace_path)
    project = ws.create_project("Resume Monitoring")
    project.description = "Text overview + drift for resumes"
    project.save()

    ws.add_report(project.id, report)
    print("Workspace populated:")
    print(f"  Workspace: {workspace_path}")
    print(f"  Project:   {project.name} ({project.id})")


if __name__ == "__main__":
    main()
