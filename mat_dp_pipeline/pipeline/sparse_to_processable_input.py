import itertools
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from mat_dp_pipeline.pipeline.common import ProcessableInput, SparseYearsInput
from mat_dp_pipeline.sdf import Year


def _interpolate_intensities(
    df: pd.DataFrame, years: list[Year], techs: list[tuple[str, str]]
) -> pd.DataFrame:
    df_index_names = df.index.names  # to restore later

    idx = pd.MultiIndex.from_tuples(
        ((year, *tech) for year, tech in itertools.product(years, techs))
    ).sort_values()

    df = df.sort_index(axis=0).sort_index(axis=1).reindex(idx)
    interpolated_array = df.unstack((1, 2)).interpolate(method="index").values.flatten()  # type: ignore

    def get_permutation():
        """Creates a permutation of indexes for conversion of `interpolated_array` to
        the flatten format of the initial `df` array. It's a bit complicated, so... you'll
        just have to trust me on this.
        """
        n_techs = len(techs)
        n_resources = len(df.columns)
        n_years = len(years)

        x, y = np.divmod(np.arange(n_techs * n_years), n_techs)
        n = (
            np.remainder(np.arange(n_techs * n_resources * n_years), n_resources)
            * n_techs
        )
        r = np.repeat(n_techs * n_resources * x + y, n_resources)
        final = n + r
        return final.astype(int)

    ordering = get_permutation()
    df = pd.DataFrame(
        interpolated_array[ordering].reshape(-1, len(df.columns)),
        index=df.index,
        columns=df.columns,
    )

    # index names were lost, restoring it
    df.index.names = df_index_names
    return df


def _interpolate_indicators(
    df: pd.DataFrame, years: list[Year], resources: list[str]
) -> pd.DataFrame:
    df_index_names = df.index.names  # to restore later

    idx = pd.MultiIndex.from_product([years, resources]).sort_values()

    df = df.sort_index(axis=0).sort_index(axis=1).reindex(idx)
    interpolated_array = df.unstack().interpolate(method="index").values.flatten()  # type: ignore

    def get_permutation():
        """Creates a permutation of indexes for conversion of `interpolated_array` to
        the flatten format of the initial `df` array. It's a bit complicated, so... you'll
        just have to trust me on this.
        """

        n_resources = len(resources)
        n_indicators = len(df.columns)
        n_years = len(years)

        x, y = np.divmod(np.arange(n_resources * n_years), n_resources)
        n = (
            np.remainder(np.arange(n_resources * n_indicators * n_years), n_indicators)
            * n_resources
        )
        r = np.repeat(n_resources * n_indicators * x + y, n_indicators)
        final = n + r
        return final.astype(int)

    ordering = get_permutation()
    df = pd.DataFrame(
        interpolated_array[ordering].reshape(-1, len(df.columns)),
        index=df.index,
        columns=df.columns,
    )

    # index names were lost, restoring it
    df.index.names = df_index_names
    return df


def to_processable_input(
    path: Path, sparse_years_input: SparseYearsInput
) -> Iterator[tuple[Path, Year, ProcessableInput]]:
    intensities = sparse_years_input.intensities
    targets = sparse_years_input.targets
    indicators = sparse_years_input.indicators

    intensities_years = list(intensities.index.get_level_values(0).unique())
    indicator_years = list(indicators.index.get_level_values(0).unique())
    target_years: list[Year] = sorted(targets.columns.astype(Year).unique().to_list())

    assert intensities_years[0] == Year(0), "No initial intensities provided!"
    assert indicator_years[0] == Year(0), "No initial indicators provided!"
    assert target_years, "No years in targets!"

    intensities_techs = intensities.droplevel(0).index.to_list()
    target_techs = targets.index.to_list()
    indicators_resources = indicators.droplevel(0).index.to_list()
    assert set(target_techs) <= set(
        intensities_techs
    ), f"Target's technologies are not a subset of intensities' techs! ({target_techs})"

    # Swap Year(0) with the first year from targets
    first_year = Year(target_years[0])
    intensities = intensities.rename({Year(0): first_year})
    indicators = indicators.rename({Year(0): first_year})

    intensities = _interpolate_intensities(intensities, target_years, target_techs)
    indicators = _interpolate_indicators(indicators, target_years, indicators_resources)

    # ProcessableInput is for a given year, so we have to proces year by year in a loop
    # We only consider target years, starting from the second one.
    assert isinstance(intensities, pd.DataFrame)
    assert isinstance(indicators, pd.DataFrame)
    for year in target_years:
        inpt = ProcessableInput(
            intensities=intensities.loc[year, :],
            targets=targets.loc[:, str(year)],
            indicators=indicators.loc[year, :],
        )
        yield path, year, inpt
