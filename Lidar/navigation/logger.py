import csv
import os

def init_logger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "x", "y", "z",
            "reward",
            "collision",
            "success",
            "attack"
        ])

def log_step(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)
