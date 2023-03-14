from io import StringIO

import pandas as pd

from mat_dp_pipeline.pipeline.calculation import calculate


def test_calculation(calculation_test_input):
    result = calculate(calculation_test_input)

    expected_emissions = pd.read_csv(
        StringIO(
            """Indicator,Category,Specific,Steel,Wood
CO2,Tool,Hammer,5.5,105.0
CO2,Tool,Pliers,110.0,2100.0
PM25,Tool,Hammer,10.5,110.0
PM25,Tool,Pliers,210.0,2200.0"""
        )
    ).rename_axis("Resource", axis="columns")

    expected_required_resources = (
        pd.DataFrame(
            data=[("Tool", "Hammer", 5, 50), ("Tool", "Pliers", 100, 1000)],
            columns=["Category", "Specific", "Steel", "Wood"],
        )
        .set_index(["Category", "Specific"])
        .rename_axis("Resource", axis="columns")
    )

    pd.testing.assert_frame_equal(
        result.emissions.reset_index(), expected_emissions.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        result.required_resources, expected_required_resources
    )
