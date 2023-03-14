from pathlib import Path

import pandas as pd

from mat_dp_pipeline.abstract_data_sources import IndicatorsSource, IntensitiesSource
from mat_dp_pipeline.data_sources.utils import location_to_path


class MatDPDBIntensitiesSource(IntensitiesSource):
    _materials_spreadsheet: Path

    def __init__(self, materials_spreadsheet: Path):
        self._materials_spreadsheet = materials_spreadsheet

    def _raw(self) -> pd.DataFrame:
        df = (
            pd.read_excel(
                self._materials_spreadsheet,
                sheet_name="Material intensities",
                header=1,
            )
            .drop(
                columns=[
                    "Total",
                    "Comments",
                    "Data collection responsible",
                    "Data collection date",
                    "Vehicle/infrastructure primary purpose",
                ]
            )
            .rename(
                columns={
                    "Technology category": "Category",
                    "Technology name": "Specific",
                    "Technology description": "Description",
                }
            )
        )
        units = df["Units"].str.split("/", n=1, expand=True)
        df.pop("Units")
        df.insert(3, "Production Unit", units.iloc[:, 1])
        df.insert(3, "Material Unit", units.iloc[:, 0])
        return df

    def __call__(self, output_dir: Path) -> None:
        ## Drop NaN based on resource value columns only
        # df = df.dropna(subset=df.columns[6:])
        for location, intensities in self._raw().groupby("Location"):
            path = location_to_path(str(location))
            df = intensities.drop(columns=["Location"])
            location_dir = output_dir / path
            location_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(location_dir / "techs.csv", index=False)


class MatDPDBIndicatorsSource(IndicatorsSource):
    _materials_spreadsheet: Path

    def __init__(self, materials_spreadsheet: Path):
        self._materials_spreadsheet = materials_spreadsheet

    def __call__(self, output_dir: Path) -> None:
        output_dir.mkdir(exist_ok=True, parents=True)
        (
            pd.read_excel(self._materials_spreadsheet, sheet_name="Material emissions")
            .drop(
                columns=[
                    "Material description",
                    "Object title in Ecoinvent",
                    "Location of dataset",
                    "Notes",
                ]
            )
            .rename(columns={"Material code": "Resource"})
            .dropna()
            .to_csv(output_dir / "indicators.csv", index=False)
        )
