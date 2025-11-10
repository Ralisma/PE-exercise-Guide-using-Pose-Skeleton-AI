import csv
from datetime import datetime
import os

class Logger:
    def __init__(self, log_file="exercise_log.csv"):
        self.log_file = log_file
        self._ensure_log_file()

    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["timestamp", "pose", "confidence", "status"])

    def log_result(self, pose, conf, status):
        try:
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow([datetime.now(), pose, f"{conf:.2f}", status])
        except Exception as e:
            print(f"Logging failed: {str(e)}")
