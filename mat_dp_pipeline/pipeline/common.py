from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import mat_dp_pipeline.sdf.standard_data_format as sdf
from mat_dp_pipeline.common import validate


@dataclass(eq=False, order=False)
class SparseYearsInput:
    """Data amalgamated from hierachical structure, with potential gaps between the years.

    The gaps between years are later filled with values (interpolated), but this isn't in scope
    of this class.

    Year is another dimension here -- level 0 index in intensities & indicators,
    year columns in targets.

    Attributes:
        intensities (DataFrame): (Year, Tech) x (Resource1, Resource2, ..., ResourceN)
        targets (DataFrame): Tech x  (Year1, Year2, ..., YearK). Tech keys must be a subset of
            intensities' keys
        indicators (DataFrame): (Year, Resource) x (Indicator1, Indicator2, ..., IndicatorM)
            There should be exactly N resources, matching columns in intensities frame
        tech_metadata (DataFrame): Technologies metadata
    """

    intensities: pd.DataFrame
    targets: pd.DataFrame
    indicators: pd.DataFrame
    tech_metadata: pd.DataFrame

    def copy(self) -> "SparseYearsInput":
        return SparseYearsInput(
            intensities=self.intensities.copy(),
            targets=self.targets.copy(),
            indicators=self.indicators.copy(),
            tech_metadata=self.tech_metadata.copy(),
        )

    def validate(self) -> None:
        """Validates whether an object represents a valid instance of
        CombinedInput.

        Raises:
            ValueError: when validation fails
        """
        sdf.validate_tech_units(self.tech_metadata)
        # TODO: reindex also intensities vs targets techs
        intensities_resources = set(self.intensities.columns)
        indicators_resources = set(self.indicators.index.get_level_values("Resource"))
        common_resources = intensities_resources & indicators_resources
        diff = intensities_resources.symmetric_difference(indicators_resources)
        if diff:
            sorted_resources = sorted(common_resources)
            self.intensities = self.intensities.reindex(columns=sorted_resources)
            self.indicators = self.indicators.reindex(
                sorted_resources, level="Resource"
            )

        # TODO: this is weird, throwing after reindex? Maybe ok? Discussion needed ;)
        validate(
            not diff,
            f"Intensities' and indicators' resources aren't matched on keys: {diff}",
        )


@dataclass(eq=False, order=False)
class ProcessableInput:
    """Single processable input. The frames in processable input do not contain year
    dimension. This is the lowest level structure, ready for processing.

    Attributes:
        intensities (DataFrame): Tech x (Resource1, Resource2, ..., ResourceN)
        targets (Series): Tech -> float. Tech keys must be a subset of intensities' keys
        indicators (DataFrame): Resource x (Indicator1, Indicator2, ..., IndicatorM)
            There should be exactly N resources, matching columns in intensities frame
    """

    intensities: pd.DataFrame
    targets: pd.Series
    indicators: pd.DataFrame

    def copy(self) -> "ProcessableInput":
        return ProcessableInput(
            intensities=self.intensities.copy(),
            targets=self.targets.copy(),
            indicators=self.indicators.copy(),
        )

    def save(self, directory: Path, exist_ok: bool = False) -> None:
        """Save ProcessableInput to files in a directory

        Args:
            directory (Path): output directory (must exist)
            exist_ok (bool, optional):  Is it OK if files exist already. They will be overriden if so. Defaults to False.
        """
        assert directory.is_dir()

        intensities_file = directory / "intensities.csv"
        targets_file = directory / "targets.csv"
        indicators_file = directory / "indicators.csv"

        if not exist_ok:
            assert (
                not intensities_file.exists()
                and not targets_file.exists()
                and not indicators_file.exists()
            )

        self.intensities.to_csv(intensities_file)
        self.targets.to_csv(targets_file)
        self.indicators.to_csv(indicators_file)
