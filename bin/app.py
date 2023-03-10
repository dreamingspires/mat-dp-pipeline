import argparse
from pathlib import Path

from mat_dp_pipeline import App, create_sdf, pipeline
from mat_dp_pipeline.data_sources import (
    StoredIndicators,
    StoredIntensities,
    StoredTargets,
)
from mat_dp_pipeline.data_sources.mat_dp_db import (
    MatDPDBIndicatorsSource,
    MatDPDBIntensitiesSource,
)
from mat_dp_pipeline.data_sources.tmba import TMBATargetsSource


def main():
    parser = argparse.ArgumentParser()
    # TODO: groups. either sdf-source or materials & targets (+their types of data source)
    parser.add_argument("--sdf-source", type=Path)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--sdf-output", type=Path)

    args = parser.parse_args()

    TARGETS_PARAMETERS = [
        "Power Generation (Aggregate)",
        "Power Generation Capacity (Aggregate)",
    ]

    if not args.sdf_source:
        # TODO: this is just a quick hacky/demo version of course
        assert args.materials and args.targets
        sdf = create_sdf(
            intensities=MatDPDBIntensitiesSource(args.materials),
            indicators=MatDPDBIndicatorsSource(args.materials),
            targets=TMBATargetsSource(args.targets, TARGETS_PARAMETERS),
        )
    else:
        sdf = create_sdf(args.sdf_source)

    if args.sdf_output:
        sdf.save(args.sdf_output)

    output = pipeline(sdf)
    App(output, ["Parameter"]).serve()


if __name__ == "__main__":
    main()
