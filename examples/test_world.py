from pathlib import Path

from mat_dp_pipeline import Pipeline

pipeline = Pipeline()

output = pipeline(Path("./World"))
