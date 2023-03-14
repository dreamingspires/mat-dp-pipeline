import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from mat_dp_pipeline.common import validate
from mat_dp_pipeline.pipeline.common import SparseYearsInput
from mat_dp_pipeline.sdf import StandardDataFormat, Year


def overlay_in_order(
    df: pd.DataFrame,
    base_overlay: pd.DataFrame,
    yearly_overlays: dict[Year, pd.DataFrame],
) -> pd.DataFrame:
    overlaid = df

    # Overlays sorted by year, first one being base_overlay (year 0)
    sorted_overlays = sorted(({Year(0): base_overlay} | yearly_overlays).items())

    for year, overlay in sorted_overlays:
        if overlay.empty:
            continue

        # Add "Year" level to the index. Concat is idiomatic way of doing it
        update_df = pd.concat({year: overlay}, names=["Year"])

        if overlaid.empty:
            overlaid = update_df.copy()
        else:
            overlaid = update_df.combine_first(overlaid)

    return overlaid


def flatten_hierarchy(
    root_sdf: StandardDataFormat,
) -> Iterator[tuple[Path, SparseYearsInput]]:
    def dfs(
        root: StandardDataFormat, inpt: SparseYearsInput, label: Path
    ) -> Iterator[tuple[Path, SparseYearsInput]]:
        validate(
            root.indicators.empty
            or inpt.indicators.empty
            or list(inpt.indicators.columns) == list(root.indicators.columns),
            f"{label}: Indicators' names on each level have to be the same!",
        )

        overlaid = inpt.copy()
        overlaid.intensities = overlay_in_order(
            overlaid.intensities, root.intensities, root.intensities_yearly
        )
        overlaid.indicators = overlay_in_order(
            overlaid.indicators, root.indicators, root.indicators_yearly
        )
        if overlaid.tech_metadata.empty:
            overlaid.tech_metadata = root.tech_metadata
        else:
            overlaid.tech_metadata = (
                pd.concat([overlaid.tech_metadata, root.tech_metadata])
                .groupby(level=(0, 1))
                .last()
            )

        # Go down in the hierarchy
        for name, directory in root.children.items():
            yield from dfs(directory, overlaid, label / name)

        # Yield only leaves
        if not root.children:
            assert root.targets is not None
            overlaid.targets = root.targets
            # Trim tech_meta to the techs specified in targets
            overlaid.tech_metadata = overlaid.tech_metadata.reindex(
                overlaid.targets.index
            )

            # TODO: add a parameter controlling whether validation yields a warning or exception or is ignored
            # Maybe group the errors and show at the end, otherwise there's a LOT of errors
            try:
                overlaid.validate()
            except ValueError as e:
                logging.error(f"Validation failed for {label}")
                logging.error(e)
                # raise e
            yield label, overlaid

    initial = SparseYearsInput(
        intensities=pd.DataFrame(),
        targets=pd.DataFrame(),
        indicators=pd.DataFrame(),
        tech_metadata=pd.DataFrame(),
    )
    yield from dfs(root_sdf, initial, Path(root_sdf.name))
