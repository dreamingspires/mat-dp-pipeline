import pandas as pd

from mat_dp_pipeline.sdf_to_input import ProcessableInput
from mat_dp_pipeline.calculation import calculate, ProcessableInput


def test_calculation(data_path):
    root = data_path("CalculationTest")

    intensities = pd.read_csv(root / "techs.csv").set_index(["Category", "Specific"])
    indicators = pd.read_csv(root / "indicators.csv").set_index("Material")
    targets = (
        pd.read_csv(root / "targets.csv").set_index(["Category", "Specific"]).iloc[:, 0]
    )

    inpt = ProcessableInput(
        intensities=intensities, targets=targets, indicators=indicators
    )

    result = calculate(inpt)
    expected_emissions = pd.DataFrame(
        data=[("Steel", 115.5, 220.5), ("Wood", 2205.0, 2310.0)],
        columns=["Resource", "CO2", "PM25"],
    ).set_index("Resource")

    expected_required_resources = pd.DataFrame(
        data=[("Tool", "Hammer", 5, 50), ("Tool", "Pliers", 100, 1000)],
        columns=["Category", "Specific", "Steel", "Wood"],
    ).set_index(["Category", "Specific"])

    pd.testing.assert_frame_equal(result.emissions, expected_emissions)
    pd.testing.assert_frame_equal(
        result.required_resources, expected_required_resources
    )
