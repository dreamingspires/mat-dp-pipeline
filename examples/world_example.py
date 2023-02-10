from pathlib import Path

from mat_dp_pipeline import pipeline

output = pipeline(Path(__file__).parent.parent / "test_data/World")
a = 1
