from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from mat_dp_pipeline.standard_data_format import StandardDataFormat

CheckpointInput = TypeVar("CheckpointInput")
ProcessableInput = TypeVar("ProcessableInput")
ProcessedOutput = TypeVar("ProcessedOutput")
Output = TypeVar("Output")


class AbstractPipeline(
    Generic[CheckpointInput, ProcessableInput, ProcessedOutput, Output], ABC
):
    @abstractmethod
    def _path_to_sdf(self, path: Path) -> StandardDataFormat:
        ...

    @abstractmethod
    def _sdf_to_checkpoints(self, sdf: StandardDataFormat) -> CheckpointInput:
        ...

    @abstractmethod
    def _checkpoints_to_processable_input(
        self, checkpoints: CheckpointInput
    ) -> ProcessableInput:
        ...

    @abstractmethod
    def _processable_to_processed(
        self, processable: ProcessableInput
    ) -> ProcessedOutput:
        ...

    @abstractmethod
    def _processed_to_output(self, processed: ProcessedOutput) -> Output:
        ...

    def __call__(self, path: Path) -> Output:
        return self._processed_to_output(
            self._processable_to_processed(
                self._checkpoints_to_processable_input(
                    self._sdf_to_checkpoints(self._path_to_sdf(path))
                )
            )
        )
