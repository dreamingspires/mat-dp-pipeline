from abc import ABC, abstractmethod
from pathlib import Path


class BaseSource(ABC):
    @abstractmethod
    def __call__(self, output_dir: Path) -> None:
        """Prepare a Standard Data Format data and save it in the `output_dir`
        Args:
            output_dir (Path): Output SDF root directory
        """
        ...


class IntensitiesSource(BaseSource):
    pass


class TargetsSource(BaseSource):
    pass


class IndicatorsSource(BaseSource):
    pass
