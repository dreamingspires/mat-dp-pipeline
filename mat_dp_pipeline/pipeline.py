from abc import ABC, abstractproperty
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    data: list[LabelledOutput]

    def __init__(self, data: list[LabelledOutput]):
        self.data = data


class DataSource(ABC):
    @abstractproperty
    def sdf(self) -> sdf.StandardDataFormat:
        ...


def pipeline(
    input_data: DataSource | Path, output_sdf_dir: Optional[Path] = None
) -> PipelineOutput:
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
