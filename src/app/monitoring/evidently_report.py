import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_data_drift_report(reference_data_path: str, current_data_path: str, report_path: str):
    """
    Generates a data drift report using Evidently AI.
    """
    reference_df = pd.read_csv(reference_data_path)
    current_df = pd.read_csv(current_data_path)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df[['Resume_str']], current_data=current_df[['Resume_str']])
    report.save_html(report_path)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    data_path = "Resume.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        reference = df.sample(frac=0.5, random_state=42)
        current = df.drop(reference.index)

       
        reference.to_csv("reference_temp.csv", index=False)
        current.to_csv("current_temp.csv", index=False)

        generate_data_drift_report("reference_temp.csv", "current_temp.csv", "data_drift_report.html")

       
        os.remove("reference_temp.csv")
        os.remove("current_temp.csv")
    else:
        print(f"Data file not found at {data_path}")