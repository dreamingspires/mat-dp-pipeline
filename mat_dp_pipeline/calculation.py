from dataclasses import dataclass
from functools import cache
import typing

import pandas as pd

from mat_dp_pipeline.sdf_to_input import ProcessableInput


@dataclass(frozen=True)
class ProcessedOutput:
    required_resources: pd.DataFrame
    emissions: pd.DataFrame

    @property
    @cache
    def indicators(self) -> set[str]:
        """Get a set of indicators stored in `emissions`.

        Returns:
            set[str]: Set of indicator names
        """
        return set(self.emissions.index.get_level_values(0).to_list())

    def emissions_for_indicator(self, indicator: str):
        return self.emissions.loc[indicator, :]

    def total_emissions(self) -> pd.Series:
        """Emissions per indicator

        Returns:
            pd.Series: Series indexed by indicators
        """
        return self.emissions.sum(axis=0)


def calculate(inpt: ProcessableInput) -> ProcessedOutput:
    required_resources = inpt.intensities.mul(inpt.targets, axis="index").rename_axis(
        index=["Category", "Specific"]
    )
    emissions_dict: dict[str, pd.Series] = {
        str(indicator): required_resources.sum(axis=0).mul(inpt.indicators[indicator])
        for indicator in inpt.indicators.columns
    }

    # Move indicators to cols
    emissions = pd.concat(emissions_dict).unstack(level=0).rename_axis(index="Resource")
    assert isinstance(emissions, pd.DataFrame)
    return ProcessedOutput(required_resources=required_resources, emissions=emissions)
