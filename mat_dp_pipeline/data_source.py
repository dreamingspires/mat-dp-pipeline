from abc import ABC, abstractmethod
from pathlib import Path

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.sdf_to_input import validate_sdf


def validate(root: Path):
    pass


class DataSource(ABC):
    @abstractmethod
    def _prepare(self) -> sdf.StandardDataFormat:
        ...

    def prepare(self, output_dir: Path) -> sdf.StandardDataFormat:
        data = self._prepare()
        data.save(output_dir)
        validate_sdf(output_dir)
        return data
