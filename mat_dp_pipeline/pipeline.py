from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.calculation import ProcessedOutput, calculate
from mat_dp_pipeline.data_source import DataSource
from mat_dp_pipeline.sdf_to_input import (
    CombinedInput,
    ProcessableInput,
    combined_to_processable_input,
    sdf_to_combined_input,
    validate_sdf,
)


@dataclass(frozen=True)
class LabelledOutput(ProcessedOutput):
    year: sdf.Year
    path: Path


class PipelineOutput:
    data: list[LabelledOutput]

    def __init__(self, data: list[LabelledOutput]):
        self.data = data


CheckpointType = tuple[Path, CombinedInput]
ProcessableFullType = tuple[Path, sdf.Year, ProcessableInput]


class Pipeline:
    _data_source: DataSource | None
    validate_sdf: bool

    def __init__(
        self,
        data_source: DataSource | None = None,
        validate_sdf: bool = True,
    ):
        self._data_source = data_source
        self.validate_sdf = validate_sdf

    def _path_to_sdf(self, path: Path) -> sdf.StandardDataFormat:
        # Prepare the data in Standard Data Format
        if self._data_source:
            # data source is being validated after convertion
            return self._data_source.prepare(path)
        else:
            validate_sdf(path)
            return sdf.load(path)

    def _sdf_to_checkpoints(
        self, sdf: sdf.StandardDataFormat
    ) -> Iterator[CheckpointType]:
        return sdf_to_combined_input(sdf)

    def _checkpoints_to_processable_input(
        self, checkpoints: Iterator[CheckpointType]
    ) -> Iterator[Iterator[ProcessableFullType]]:
        for path, combined in checkpoints:
            yield combined_to_processable_input(path, combined)

    def _processable_to_processed(
        self, processable: Iterator[Iterator[ProcessableFullType]]
    ) -> Iterator[LabelledOutput]:
        for item in processable:
            for path, year, inpt in item:
                result = calculate(inpt)
                yield LabelledOutput(
                    required_resources=result.required_resources,
                    emissions=result.emissions,
                    year=year,
                    path=path,
                )

    def _processed_to_output(
        self, processed: Iterator[LabelledOutput]
    ) -> PipelineOutput:
        return PipelineOutput(list(processed))

    def __call__(self, path: Path) -> PipelineOutput:
        return self._processed_to_output(
            self._processable_to_processed(
                self._checkpoints_to_processable_input(
                    self._sdf_to_checkpoints(self._path_to_sdf(path))
                )
            )
        )
