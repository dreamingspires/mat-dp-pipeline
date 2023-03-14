import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mat_dp_pipeline.pipeline.common import ProcessableInput


@dataclass(frozen=True, order=False, eq=False)
class ProcessedOutput:
    required_resources: pd.DataFrame
    emissions: pd.DataFrame

    @property
    def indicators(self) -> set[str]:
        """Get a set of indicators stored in `emissions`.

        Returns:
            set[str]: Set of indicator names
        """
        return set(self.emissions.index.get_level_values(0).to_list())

    def emissions_for_indicator(self, indicator: str):
        return self.emissions.loc[indicator]


def calculate(inpt: ProcessableInput) -> ProcessedOutput:
    required_resources = inpt.intensities.mul(inpt.targets, axis="index").rename_axis(
        index=["Category", "Specific"], columns=["Resource"]
    )

    index = pd.MultiIndex.from_tuples(
        (
            (ind, *tech)
            for ind, tech in itertools.product(
                inpt.indicators.columns, inpt.intensities.index
            )
        ),
        names=["Indicator", "Category", "Specific"],
    )
    emissions = pd.DataFrame(
        np.einsum(
            "ij,jk->kij", required_resources.values, inpt.indicators.values
        ).reshape(len(index), -1),
        index=index,
        columns=inpt.intensities.columns,
    ).rename_axis(columns="Resource")

    assert isinstance(emissions, pd.DataFrame)
    return ProcessedOutput(required_resources=required_resources, emissions=emissions)
