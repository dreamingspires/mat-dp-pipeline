import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

import mat_dp_pipeline.standard_data_format as sdf


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


def overlay_in_order(
    df: pd.DataFrame,
    base_overlay: pd.DataFrame,
    yearly_overlays: dict[sdf.Year, pd.DataFrame],
) -> pd.DataFrame:
    overlayed = df

    base_keys = set(base_overlay.index.to_list())
    # Overlays sorted by year, first one being base_overlay (year 0)
    sorted_overlays = sorted(({sdf.Year(0): base_overlay} | yearly_overlays).items())

    for year, overlay in sorted_overlays:
        if overlay.empty:
            continue
        assert (
            set(overlay.index.to_list()) <= base_keys
        ), "Yearly file cannot introduce new items!"

        # Add "Year" level to the index. Concat is idiomatic way of doing it
        update_df = pd.concat({year: overlay}, names=["Year"])

        if overlayed.empty:
            overlayed = update_df.copy()
        else:
            overlayed = update_df.combine_first(overlayed)

    return overlayed


def sdf_to_combined_input(
    root_sdf: sdf.StandardDataFormat,
) -> Iterator[tuple[Path, CombinedInput]]:
    def dfs(
        root: sdf.StandardDataFormat, inpt: CombinedInput, label: Path
    ) -> Iterator[tuple[Path, CombinedInput]]:
        overlayed = inpt.copy()
        overlayed.intensities = overlay_in_order(
            overlayed.intensities, root.intensities, root.intensities_yearly
        )
        overlayed.indicators = overlay_in_order(
            overlayed.indicators, root.indicators, root.indicators_yearly
        )

        # Go down in the hierarchy
        for name, directory in root.children.items():
            yield from dfs(directory, overlayed, label / name)

        # Yield only leaves
        if not root.children:
            assert root.targets is not None
            overlayed.targets = root.targets
            assert overlayed.validate()
            yield label, overlayed

    initial = CombinedInput(
        intensities=pd.DataFrame(),
        targets=pd.DataFrame(),
        indicators=pd.DataFrame(),
    )
    yield from dfs(root_sdf, initial, Path(root_sdf.name))


def combined_to_processable_input(
    path: Path, combined: CombinedInput
) -> Iterator[tuple[Path, sdf.Year, ProcessableInput]]:
    intensities = combined.intensities
    targets = combined.targets
    indicators = combined.indicators

    intensities_years = list(intensities.index.get_level_values(0).unique())
    indicator_years = list(indicators.index.get_level_values(0).unique())
    target_years: list[sdf.Year] = sorted(
        targets.columns.astype(sdf.Year).unique().to_list()
    )

    assert intensities_years[0] == sdf.Year(0), "No initial intensities provided!"
    assert indicator_years[0] == sdf.Year(0), "No initial indicators provided!"
    assert target_years, "No years in targets!"

    intensities_techs = intensities.droplevel(0).index.to_list()
    target_techs = targets.index.to_list()
    indicators_resources = indicators.droplevel(0).index.to_list()
    assert set(target_techs) <= set(
        intensities_techs
    ), "Target's technologies are not a subset of intensities' techs!"

    # Swap Year(0) with the first year from targets
    first_year = sdf.Year(target_years[0])
    intensities = intensities.rename({sdf.Year(0): first_year})
    indicators = indicators.rename({sdf.Year(0): first_year})

    tech_year_idx = pd.MultiIndex.from_tuples(
        ((year, *tech) for year, tech in itertools.product(target_years, target_techs))
    )
    resource_year_idx = pd.MultiIndex.from_product([target_years, indicators_resources])

    intensities: pd.DataFrame = (
        intensities.reindex(tech_year_idx)
        .unstack()
        .unstack()  # Leave only year in the index
        .interpolate(method="index")
        .stack()  # type: ignore
        .stack()
    )
    indicators: pd.DataFrame = (
        indicators.reindex(resource_year_idx)
        .unstack()  # Leave only year in the index
        .interpolate(method="index")
        .stack()  # type: ignore
    )

    # ProcessableInput is for a given year, so we have to proces year by year in a loop
    # We only consider target years, starting from the second one.
    assert isinstance(intensities, pd.DataFrame)
    assert isinstance(indicators, pd.DataFrame)
    for year in target_years:
        inpt = ProcessableInput(
            intensities=intensities.loc[year, :].sort_index(),
            targets=targets.loc[:, str(year)].sort_index(),
            indicators=indicators.loc[year, :].sort_index(),
        )
        yield path, year, inpt


def sdf_to_processable_input(
    root: sdf.StandardDataFormat,
) -> Iterator[tuple[Path, sdf.Year, ProcessableInput]]:
    for path, combined in sdf_to_combined_input(root):
        yield from combined_to_processable_input(path, combined)
