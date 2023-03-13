import tempfile
from pathlib import Path
from typing import overload

import mat_dp_pipeline.data_sources.definitions as ds
from mat_dp_pipeline.sdf.standard_data_format import StandardDataFormat, load


@overload
def create_sdf(
    *,
    intensities: ds.IntensitiesSource,
    indicators: ds.IndicatorsSource,
    targets: ds.TargetsSource | list[ds.TargetsSource],
) -> StandardDataFormat:
    ...


@overload
def create_sdf(source: Path | str) -> StandardDataFormat:
    ...


def create_sdf(
    source: Path | str | None = None,
    *,
    intensities: ds.IntensitiesSource | None = None,
    indicators: ds.IndicatorsSource | None = None,
    targets: ds.TargetsSource | list[ds.TargetsSource] | None = None,
) -> StandardDataFormat:
    if source:
        return load(Path(source))
    else:
        assert intensities and indicators and targets
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)

            targets_list = targets if isinstance(targets, list) else [targets]
            for t in targets_list:
                t(path)
            intensities(path)
            indicators(path)

            return load(path)
