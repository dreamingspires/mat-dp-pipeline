import argparse
from pathlib import Path

from mat_dp_pipeline import App, pipeline
from mat_dp_pipeline.data_sources.mat_dp_db import MatDpDB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf-output-dir", type=Path)
    parser.add_argument("--materials", type=Path)
    parser.add_argument("--targets", type=Path)

    args = parser.parse_args()

    # path = Path(__file__).parent.parent / "test_data/World"
    # output = pipeline(path)

    # MATERIALS_EXCEL = "scratch/Material_intensities_database.xlsx"
    # TARGETS_CSV = "scratch/results_1.5deg.csv"
    TARGETS_PARAMETERS = [
        "Power Generation (Aggregate)",
        "Power Generation Capacity (Aggregate)",
    ]

    ds = MatDpDB(args.materials, args.targets, TARGETS_PARAMETERS)
    output = pipeline(ds, output_path=args.sdf_output_dir)
    App(output, ["Parameter"]).serve()


if __name__ == "__main__":
    main()
