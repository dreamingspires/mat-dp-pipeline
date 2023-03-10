import re
import shutil
from pathlib import Path

import mat_dp_pipeline.standard_data_format as sdf
from mat_dp_pipeline.data_sources.definitions import (
    IndicatorsSource,
    IntensitiesSource,
    TargetsSource,
)


def copy_files(src: Path, dst: Path, file_pattern: re.Pattern) -> None:
    """Copy files from `src` to `dst` matching `file_pattern`. The operation
    preserves relative paths of the copied files.
    """

    source_files = filter(
        lambda f: f.is_file() and file_pattern.match(f.name), src.rglob("*")
    )
    for f in source_files:
        relative = f.relative_to(src)
        dst_file = dst / relative
        if dst_file.exists():
            raise ValueError(f"File {dst_file} already exists!")
        dst_file.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(f, dst_file)


class _StoredSource:
    path: Path

    def __init__(self, path: Path):
        self.path = path


class StoredTargets(_StoredSource, TargetsSource):
    def __call__(self, output_dir: Path) -> None:
        # TODO: better would be to change *Reader.file_pattern into a static member
        pattern = sdf.TargetsReader().file_pattern
        copy_files(self.path, output_dir, pattern)


class StoredIntensities(_StoredSource, IntensitiesSource):
    def __call__(self, output_dir: Path) -> None:
        pattern = sdf.IntensitiesReader().file_pattern
        copy_files(self.path, output_dir, pattern)


class StoredIndicators(_StoredSource, IndicatorsSource):
    def __call__(self, output_dir: Path) -> None:
        pattern = sdf.IndicatorsReader().file_pattern
        copy_files(self.path, output_dir, pattern)
