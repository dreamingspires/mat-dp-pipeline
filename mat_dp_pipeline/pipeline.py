import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, overload

import pandas as pd

import mat_dp_pipeline.data_sources as ds
from mat_dp_pipeline.calculation import ProcessedOutput, calculate
from mat_dp_pipeline.sdf_to_input import flatten_hierarchy, to_processable_input
from mat_dp_pipeline.standard_data_format import StandardDataFormat, Year
from mat_dp_pipeline.standard_data_format import load as load_sdf


@dataclass(frozen=True)
class LabelledOutput(ProcessedOutput):
    year: Year
    path: Path


class PipelineOutput:
    """The processed data from the pipeline in an easily accessible form."""

    _by_year: dict[Year, dict[Path, LabelledOutput]]
    _by_path: dict[Path, dict[Year, LabelledOutput]]
    _length: int
    _indicators: set[str]
    _tech_metadata: pd.DataFrame

    def __init__(self, data: list[LabelledOutput], tech_metadata: pd.DataFrame):
        self._by_year = defaultdict(dict)
        self._by_path = defaultdict(dict)
        self._tech_metadata = tech_metadata

        if data:
            # We know from the computation that each LabelledOutput has the same
            # set of indicators, so we'll just take the first one
            # TODO: we must assert somewhere (not here?) that all the outputs have the same indicators
            self._indicators = data[0].indicators
        else:
            self._indicators = set()

        for output in data:
            self._by_year[output.year][output.path] = output
            self._by_path[output.path][output.year] = output
        self._length = len(data)

    def emissions(self, key: Path | str, indicator: str) -> pd.DataFrame:
        """TODO:

        Args:
            indicator (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        data = self[Path(key)]
        return pd.concat(
            {k: v.emissions.loc[indicator, :] for k, v in data.items()}, names=["Year"]
        )

    def resources(self, key: Path | str) -> pd.DataFrame:
        data = self[Path(key)]
        return pd.concat(
            {k: v.required_resources for k, v in data.items()}, names=["Year"]
        )

    @property
    def by_year(self) -> dict[Year, dict[Path, LabelledOutput]]:
        return self._by_year

    @property
    def by_path(self) -> dict[Path, dict[Year, LabelledOutput]]:
        return self._by_path

    def keys(self, axis: type):
        assert axis in (Year, Path)
        if axis == Year:
            return self.by_year.keys()
        else:
            return self.by_path.keys()

    @property
    def indicators(self):
        return self._indicators

    @property
    def tech_metadata(self):
        return self._tech_metadata

    @overload
    def __getitem__(self, key: Year) -> dict[Path, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: Path | str) -> dict[Year, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: tuple[Year, Path | str]) -> LabelledOutput:
        ...

    @overload
    def __getitem__(self, key: tuple[Path | str, Year]) -> LabelledOutput:
        ...

    def __getitem__(
        self,
        key: Year | Path | str | tuple[Year, Path | str] | tuple[Path | str, Year],
    ) -> dict[Path, LabelledOutput] | dict[Year, LabelledOutput] | LabelledOutput:
        if isinstance(key, Year):
            return self.by_year[key]
        elif isinstance(key, (Path, str)):
            return self.by_path[Path(key)]
        elif isinstance(key, tuple):
            assert len(key) == 2
            year, path = key
            if isinstance(year, (Path, str)):
                path, year = year, path
            assert isinstance(year, Year) and isinstance(path, (Path, str))
            return self.by_year[year][Path(path)]

    def __iter__(self) -> Iterator[LabelledOutput]:
        for _, d in self.by_year.items():
            yield from d.values()

    def __len__(self) -> int:
        return self._length


@overload
def create_sdf(
    *,
    intensities: ds.IntensitiesSource,
    indicators: ds.IndicatorsSource,
    targets: ds.TargetsSource | list[ds.TargetsSource],
) -> StandardDataFormat:
    ...


@overload
def create_sdf(source: Path | str) -> StandardDataFormat:
    ...


def create_sdf(
    source: Path | str | None = None,
    *,
    intensities: ds.IntensitiesSource | None = None,
    indicators: ds.IndicatorsSource | None = None,
    targets: ds.TargetsSource | list[ds.TargetsSource] | None = None,
) -> StandardDataFormat:
    if source:
        return load_sdf(Path(source))
    else:
        assert intensities and indicators and targets
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)

            targets_list = targets if isinstance(targets, list) else [targets]
            for t in targets_list:
                t(path)
            intensities(path)
            indicators(path)

            return load_sdf(path)


def pipeline(sdf: StandardDataFormat) -> PipelineOutput:
    """Converts the input data to the PipelineOutput format.

    Returns:
        PipelineOutput: The fully converted output of the pipeline
    """
    processed = []
    tech_metadata = pd.DataFrame()
    for path, sparse_years in flatten_hierarchy(sdf):
        tech_metadata = (
            pd.concat([sparse_years.tech_metadata, tech_metadata])
            .groupby(level=(0, 1))
            .last()
        )
        for path, year, inpt in to_processable_input(path, sparse_years):
            result = calculate(inpt)
            processed.append(
                LabelledOutput(
                    required_resources=result.required_resources,
                    emissions=result.emissions,
                    year=year,
                    path=path,
                )
            )
    return PipelineOutput(processed, tech_metadata)
