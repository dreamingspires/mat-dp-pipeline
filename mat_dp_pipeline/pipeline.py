import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, overload

import pandas as pd

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.calculation import ProcessedOutput, calculate
from mat_dp_pipeline.sdf_to_input import (
    combined_to_processable_input,
    sdf_to_combined_input,
)


@dataclass(frozen=True)
class LabelledOutput(ProcessedOutput):
    year: sdf.Year
    path: Path


class PipelineOutput:
    """
    The processed data from the pipeline in an easily accessible form.
    """
    _by_year: dict[sdf.Year, dict[Path, LabelledOutput]]
    _by_path: dict[Path, dict[sdf.Year, LabelledOutput]]
    _length: int
    _indicators: set[str]
    _tech_meta: pd.DataFrame

    def __init__(self, data: list[LabelledOutput], tech_meta: pd.DataFrame):
        self._by_year = defaultdict(dict)
        self._by_path = defaultdict(dict)
        self._tech_meta = tech_meta

        if data:
            # We know from the computation that each LabelledOutput has the same
            # set of indicators, so we'll just take the first one
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
    def by_year(self) -> dict[sdf.Year, dict[Path, LabelledOutput]]:
        return self._by_year

    @property
    def by_path(self) -> dict[Path, dict[sdf.Year, LabelledOutput]]:
        return self._by_path

    def keys(self, axis: type):
        assert axis in (sdf.Year, Path)
        if axis == sdf.Year:
            return self.by_year.keys()
        else:
            return self.by_path.keys()

    @property
    def indicators(self):
        return self._indicators

    @property
    def tech_meta(self):
        return self._tech_meta

    @overload
    def __getitem__(self, key: sdf.Year) -> dict[Path, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: Path | str) -> dict[sdf.Year, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: tuple[sdf.Year, Path | str]) -> LabelledOutput:
        ...

    @overload
    def __getitem__(self, key: tuple[Path | str, sdf.Year]) -> LabelledOutput:
        ...

    def __getitem__(
        self,
        key: sdf.Year
        | Path
        | str
        | tuple[sdf.Year, Path | str]
        | tuple[Path | str, sdf.Year],
    ) -> dict[Path, LabelledOutput] | dict[sdf.Year, LabelledOutput] | LabelledOutput:
        if isinstance(key, sdf.Year):
            return self.by_year[key]
        elif isinstance(key, (Path, str)):
            return self.by_path[Path(key)]
        elif isinstance(key, tuple):
            assert len(key) == 2
            year, path = key
            if isinstance(year, (Path, str)):
                path, year = year, path
            assert isinstance(year, sdf.Year) and isinstance(path, (Path, str))
            return self.by_year[year][Path(path)]

    def __iter__(self) -> Iterator[LabelledOutput]:
        for _, d in self.by_year.items():
            yield from d.values()

    def __len__(self) -> int:
        return self._length


class DataSource(ABC):
    """
    A custom data source format, that has the property sdf. This property must
    return the data source in the standard data format.
    """
    @abstractmethod
    def __call__(self, output_dir: Path) -> None:
        """Prepare a Standard Data Format data and save it in the `output_dir`

        Args:
            output_dir (Path): Output SDF root directory
        """
        ...


@overload
def pipeline(source: Path) -> PipelineOutput:
    ...


@overload
def pipeline(source: DataSource, output_path: Path | None = None) -> PipelineOutput:
    ...


def pipeline(
    source: Path | DataSource | None = None,
    output_path: Path | None = None,
) -> PipelineOutput:
    """
    Converts the input data to the PipelineOutput format.

    Args:
        input_data (DataSource | Path): Receives either a custom data source or a
            path to a fully described standard data format
        output_sdf_dir (Optional[Path], optional): If provided, outputs the standard
            data format version of the input_data to this directory. Defaults to None.

    Returns:
        PipelineOutput: The fully converted output of the pipeline
    """
    if isinstance(source, DataSource):
        if output_path:
            source(output_path)
            out_sdf = sdf.load(Path(output_path))
        else:
            with tempfile.TemporaryDirectory() as dirpath:
                source(Path(dirpath))
                out_sdf = sdf.load(Path(dirpath))
    else:
        assert isinstance(source, Path)
        out_sdf = sdf.load(source)

    processed = []
    tech_meta = pd.DataFrame()
    for path, combined in sdf_to_combined_input(out_sdf):
        tech_meta = (
            pd.concat([combined.tech_meta, tech_meta]).groupby(level=(0, 1)).last()
        )
        for path, year, inpt in combined_to_processable_input(path, combined):
            result = calculate(inpt)
            processed.append(
                LabelledOutput(
                    required_resources=result.required_resources,
                    emissions=result.emissions,
                    year=year,
                    path=path,
                )
            )
    return PipelineOutput(processed, tech_meta)
