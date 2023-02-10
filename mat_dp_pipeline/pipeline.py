from abc import abstractproperty
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.calculation import ProcessedOutput, calculate
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


class DataSource:
    @abstractproperty
    def sdf(self) -> sdf.StandardDataFormat:
        ...


CheckpointType = tuple[Path, CombinedInput]
ProcessableFullType = tuple[Path, sdf.Year, ProcessableInput]


class Pipeline:
    validate_sdf: bool

    def __init__(
        self,
        validate_sdf: bool = True,
    ):
        self.validate_sdf = validate_sdf

    def _input_to_sdf(
        self, input_data: DataSource | Path, output_sdf_dir: Optional[Path] = None
    ) -> sdf.StandardDataFormat:
        # Prepare the data in Standard Data Format
        if isinstance(input_data, Path):
            if self.validate_sdf:
                validate_sdf(input_data)
            out_sdf = sdf.load(input_data)
        else:
            out_sdf = input_data.sdf
        if output_sdf_dir:
            out_sdf.save(output_sdf_dir)
        return out_sdf

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

    def __call__(
        self, input_data: DataSource | Path, output_sdf_dir: Optional[Path] = None
    ) -> PipelineOutput:
        return self._processed_to_output(
            self._processable_to_processed(
                self._checkpoints_to_processable_input(
                    self._sdf_to_checkpoints(
                        self._input_to_sdf(input_data, output_sdf_dir)
                    )
                )
            )
        )
