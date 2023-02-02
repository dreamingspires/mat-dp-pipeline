from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import Iterator

import pandas as pd

import mat_dp_pipeline.standard_data_format as sdf

Year = int


@dataclass
class CombinedInput:
    """Input combined from hierachical structure. This is *not* year-level input.

    Year is another dimension here -- level 0 index in intensities & indicators,
    year columns in targets.

    Attributes:
        intensities (DataFrame): (Year, Tech) x (Resource1, Resource2, ..., ResourceN)
        targets (DataFrame): Tech x  (Year1, Year2, ..., YearK). Tech keys must be a subset of
            intensities' keys
        indicators (DataFrame): (Year, Resource) x (Indicator1, Indicator2, ..., IndicatorM)
            There should be exactly N resources, matching columns in intensities frame
    """

    intensities: pd.DataFrame
    targets: pd.DataFrame
    indicators: pd.DataFrame

    def copy(self) -> "CombinedInput":
        return CombinedInput(
            intensities=self.intensities.copy(),
            targets=self.targets.copy(),
            indicators=self.indicators.copy(),
        )

    def validate(self) -> bool:
        """Validates whether an object represents a valid instance of
        CombinedInput.

        Returns:
            bool: True if valid, False otherwise

        Raises:
            ValueError: when validation fails
        """
        # TODO:
        return True


@dataclass
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


def overlay_with_files(
    df: pd.DataFrame, dir: Path, reader: sdf.InputReader, by_year: bool = True
) -> pd.DataFrame:
    """Overlay `df` with data from the files in `dir`"""
    assert dir.is_dir()
    overlayed = df
    for file in filter(lambda f: f.is_file(), dir.iterdir()):
        if not (match := reader.file_pattern.match(file.name)):
            continue
        read_df = reader.read(file)

        if by_year:
            year = match.group(1) or Year(0)  # 0 -- initial
            year = Year(year)
            # Add "Year" level to the index. Concat is idiomatic way of doing it
            update_df = pd.concat({year: read_df}, names=["Year"])
        else:
            update_df = read_df

        if overlayed.empty:
            overlayed = update_df.copy()
        else:
            overlayed = update_df.combine_first(overlayed)

    return overlayed


def sdf_to_combined_input(root_dir: Path) -> Iterator[tuple[Path, CombinedInput]]:
    assert root_dir.is_dir(), "Root must be a directory!"

    def dfs(
        root: Path, inpt: CombinedInput, label: Path
    ) -> Iterator[tuple[Path, CombinedInput]]:
        sub_directories = list(filter(lambda p: p.is_dir(), root.iterdir()))

        overlayed = inpt.copy()
        overlayed.intensities = overlay_with_files(
            overlayed.intensities, root, sdf.IntensitiesReader()
        )
        overlayed.indicators = overlay_with_files(
            overlayed.indicators, root, sdf.IndicatorsReader()
        )
        overlayed.targets = overlay_with_files(
            overlayed.targets, root, sdf.TargetsReader(), by_year=False
        )

        # Go down in the hierarchy
        for dir in sub_directories:
            yield from dfs(dir, overlayed, label / dir.name)

        # Yield only leaves
        if not sub_directories:
            assert overlayed.validate()
            yield label, overlayed

    initial = CombinedInput(
        intensities=pd.DataFrame(),
        targets=pd.DataFrame(),
        indicators=pd.DataFrame(),
    )
    yield from dfs(root_dir, initial, Path(root_dir.name))


def sdf_to_processable_input(
    root_dir: Path,
) -> Iterator[tuple[Path, Year, ProcessableInput]]:
    for path, combined in sdf_to_combined_input(root_dir):
        intensities = combined.intensities
        targets = combined.targets
        indicators = combined.indicators

        intensities_years = list(intensities.index.get_level_values(0).unique())
        indicator_years = list(indicators.index.get_level_values(0).unique())
        target_years: list[Year] = sorted(
            targets.columns.astype(Year).unique().to_list()
        )

        assert intensities_years[0] == Year(0), "No initial intensities provided!"
        assert indicator_years[0] == Year(0), "No initial indicators provided!"
        assert target_years, "No years in targets!"

        intensities_techs = intensities.droplevel(0).index.to_list()
        target_techs = targets.index.to_list()
        indicators_resources = indicators.droplevel(0).index.to_list()
        assert set(target_techs) <= set(
            intensities_techs
        ), "Target's technologies are not a subset of intensities' techs!"

        # Swap Year(0) with the first year from targets
        first_year = Year(target_years[0])
        intensities = intensities.rename({Year(0): first_year})
        indicators = indicators.rename({Year(0): first_year})

        tech_year_idx = pd.MultiIndex.from_tuples(
            (
                (year, *tech)
                for year, tech in itertools.product(target_years, target_techs)
            )
        )
        resource_year_idx = pd.MultiIndex.from_product(
            [target_years, indicators_resources]
        )

        intensities = (
            intensities.reindex(tech_year_idx)
            .unstack()
            .unstack()  # Leave only year in the index
            .interpolate(method="index")
            .stack()
            .stack()
        )
        indicators = (
            indicators.reindex(resource_year_idx)
            .unstack()  # Leave only year in the index
            .interpolate(method="index")
            .stack()
        )

        # ProcessableInput is for a given year, so we have to proces year by year in a loop
        # We only consider target years, starting from the second one.
        for year in target_years:
            inpt = ProcessableInput(
                intensities=intensities.loc[year, :],
                targets=targets.loc[:, str(year)],
                indicators=indicators.loc[year, :],
            )
            yield path, year, inpt
