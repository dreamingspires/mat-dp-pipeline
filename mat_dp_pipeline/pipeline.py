from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.calculation import ProcessedOutput, calculate
from mat_dp_pipeline.data_source import DataSource
from mat_dp_pipeline.sdf_to_input import (
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


PipelineCallback = Callable[[], None]
ProcessableInputHook = Callable[[ProcessableInput], ProcessableInput]


class Pipeline:
    _sdf_root: Path
    _data_source: DataSource | None
    validate_sdf: bool
    processable_input_hook: ProcessableInputHook | None

    def __init__(
        self,
        sdf_root: Path,
        data_source: DataSource | None = None,
        validate_sdf: bool = True,
        processable_input_hook: ProcessableInputHook | None = None,
    ):
        self._sdf_root = sdf_root
        self._data_source = data_source
        self.validate_sdf = validate_sdf
        self.processable_input_hook = processable_input_hook

    def _prepare_sdf(self) -> sdf.StandardDataFormat:
        # Prepare the data in Standard Data Format
        if self._data_source:
            # data source is being validated after convertion
            return self._data_source.prepare(self._sdf_root)
        else:
            validate_sdf(self._sdf_root)
            return sdf.load(self._sdf_root)

    def __iter__(self) -> Iterator[LabelledOutput]:
        data = self._prepare_sdf()

        for path, combined in sdf_to_combined_input(data):
            for path, year, inpt in combined_to_processable_input(path, combined):
                if self.processable_input_hook:
                    inpt = self.processable_input_hook(inpt)
                result = calculate(inpt)
                yield LabelledOutput(
                    required_resources=result.required_resources,
                    emissions=result.emissions,
                    year=year,
                    path=path,
                )

    def process(
        self,
        *,
        prepared_sdf_callback: PipelineCallback | None = None,
        combined_callback: PipelineCallback | None = None,
    ) -> PipelineOutput:
        data = self._prepare_sdf()

        if prepared_sdf_callback:
            prepared_sdf_callback()

        combined_inputs = sdf_to_combined_input(data)
        if combined_callback:
            combined = list(combined_inputs)
            combined_callback()

        outputs: list[LabelledOutput] = []
        for path, combined in combined_inputs:
            for path, year, inpt in combined_to_processable_input(path, combined):
                if self.processable_input_hook:
                    inpt = self.processable_input_hook(inpt)
                result = calculate(inpt)
                outputs.append(
                    LabelledOutput(
                        required_resources=result.required_resources,
                        emissions=result.emissions,
                        year=year,
                        path=path,
                    )
                )

        return PipelineOutput(outputs)
