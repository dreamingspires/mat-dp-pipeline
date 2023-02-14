from abc import ABC, abstractproperty
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, overload

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

    def __init__(self, data: list[LabelledOutput]):
        self._by_year = defaultdict(dict)
        self._by_path = defaultdict(dict)

        for output in data:
            self._by_year[output.year][output.path] = output
            self._by_path[output.path][output.year] = output
        self._length = len(data)

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

    @overload
    def __getitem__(self, key: sdf.Year) -> dict[Path, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: Path) -> dict[sdf.Year, LabelledOutput]:
        ...

    @overload
    def __getitem__(self, key: tuple[sdf.Year, Path]) -> LabelledOutput:
        ...

    @overload
    def __getitem__(self, key: tuple[Path, sdf.Year]) -> LabelledOutput:
        ...

    def __getitem__(
        self, key: sdf.Year | Path | tuple[sdf.Year, Path] | tuple[Path, sdf.Year]
    ) -> dict[Path, LabelledOutput] | dict[sdf.Year, LabelledOutput] | LabelledOutput:
        if isinstance(key, sdf.Year):
            return self.by_year[key]
        elif isinstance(key, Path):
            return self.by_path[key]
        elif isinstance(key, tuple):
            assert len(key) == 2
            year, path = key
            if isinstance(year, Path):
                path, year = year, path
            assert isinstance(year, sdf.Year) and isinstance(path, Path)
            return self.by_year[year][path]

    def __iter__(self) -> Iterator[LabelledOutput]:
        for year, d in self.by_year.items():
            yield from d.values()

    def __len__(self) -> int:
        return self._length


class DataSource(ABC):
    """
    A custom data source format, that has the property sdf. This property must
    return the data source in the standard data format.
    """
    @abstractproperty
    def sdf(self) -> sdf.StandardDataFormat:
        ...


def pipeline(
    input_data: DataSource | Path, output_sdf_dir: Optional[Path] = None
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
    if isinstance(input_data, Path):
        out_sdf = sdf.load(input_data)
    else:
        out_sdf = input_data.sdf
    if output_sdf_dir:
        out_sdf.save(output_sdf_dir)
    processed = []
    for path, combined in sdf_to_combined_input(out_sdf):
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
    return PipelineOutput(processed)
