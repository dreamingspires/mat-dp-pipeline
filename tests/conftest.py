from pathlib import Path

import pandas as pd
import pytest

from mat_dp_pipeline.pipeline.sdf_to_input import ProcessableInput


@pytest.fixture()
def data_path():
    def inner(suffix: Path | None = None) -> Path:
        p = Path(__file__).parent.parent / "test_data" / (suffix or "")
        print(p)
        return p

    return inner


@pytest.fixture()
def calculation_test_input(data_path) -> ProcessableInput:
    root = data_path("CalculationTest")

    intensities = pd.read_csv(root / "techs.csv").set_index(["Category", "Specific"])
    indicators = pd.read_csv(root / "indicators.csv").set_index("Resource")
    targets = (
        pd.read_csv(root / "targets.csv").set_index(["Category", "Specific"]).iloc[:, 0]
    )

    return ProcessableInput(
        intensities=intensities, targets=targets, indicators=indicators
    )
