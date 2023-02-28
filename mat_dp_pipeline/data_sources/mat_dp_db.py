import itertools
from pathlib import Path
from typing import Iterator

import pandas as pd

from mat_dp_pipeline.pipeline import DataSource
import mat_dp_pipeline.standard_data_format as sdf

COUNTRY_MAPPING_CSV = Path(__file__).parent / "country_codes.csv"

# From targets' "variable" to intensities "specific" name(s)
VARIABLE_TO_SPECIFIC: dict[str, str | None] = {
    "Biomass with ccs": "Biomass + CCS",
    "Coal with ccs": "Coal + CCS",
    "Gas with ccs": "Gas + CCS",
    "Hydro": "Hydro (medium)",
    "Wind": "Offshore wind",
    "power_trade": None,  # Don't keep it
}

PARAMETER_TO_CATEGORY = {
    "Power Generation (Aggregate)": "Power plant",
}


def default_location_mapping() -> dict[str, Path]:
    countries = pd.read_csv(COUNTRY_MAPPING_CSV)
    # replace NaNs with Nones
    countries = countries.where(pd.notnull(countries), None)

    def by_col(col):
        return dict(
            zip(
                countries[col],
                (
                    Path(row["region"]) / row["name"]
                    for (_, row) in countries.iterrows()
                    if row["region"] is not None and row["name"] is not None
                ),
            )
        )

    from_matdp_region = {
        "Africa": Path("Africa"),
        "Europe": Path("Europe"),
        "Middle East and Central Asia": Path("Asia"),
        "South and East Asia": Path("Asia"),
        "Central and South America": Path("America"),
        "Oceania": Path("Oceania"),
        "North America": Path("America"),
        "Central and South America": Path("America"),
        "General": Path("."),
        "NM": Path("Africa") / "Namibia",
    }
    return by_col("alpha-2") | by_col("name") | by_col("alpha-3") | from_matdp_region


def empty_sdf(name: str) -> sdf.StandardDataFormat:
    return sdf.StandardDataFormat(
        name=name,
        intensities=pd.DataFrame(),
        intensities_yearly={},
        indicators=pd.DataFrame(),
        indicators_yearly={},
        targets=None,
        children={},
    )


class MatDpDB(DataSource):

    _materials_spreadsheet: Path
    _targets_csv: Path
    _targets_parameter: str
    _parameter_to_category: dict[str, str]
    _variable_to_specific: dict[str, str | None]
    _location_mapping: dict[str, Path]

    def __init__(
        self,
        materials_spreadsheet: Path,
        targets_csv: Path,
        targets_parameter: str,
        parameter_to_category: dict[str, str] | None = None,
        variable_to_specific: dict[str, str | None] | None = None,
        location_mapping: dict[str, Path] | None = None,
    ):
        self._materials_spreadsheet = materials_spreadsheet
        self._targets_csv = targets_csv
        self._targets_parameter = targets_parameter
        self._parameter_to_category = (
            parameter_to_category if parameter_to_category else PARAMETER_TO_CATEGORY
        )
        self._variable_to_specific = (
            variable_to_specific if variable_to_specific else VARIABLE_TO_SPECIFIC
        )
        if location_mapping is not None:
            self._location_mapping = location_mapping
        else:
            self._location_mapping = default_location_mapping()

    def _intensities(self) -> Iterator[tuple[str, pd.DataFrame]]:
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

        # Drop NaN based on resource value columns only
        df = df.dropna(subset=df.columns[6:])
        for location, intensities in df.groupby("Location"):
            yield str(location), intensities.drop(columns=["Location"])

    def _indicators(self) -> pd.DataFrame:
        return (
            pd.read_excel(self._materials_spreadsheet, sheet_name="Material emissions")
            .drop(
                columns=[
                    "Material description",
                    "Object title in Ecoinvent",
                    "Location of dataset",
                    "Notes",
                ]
            )
            .rename(columns={"Material code": "Material"})
            .dropna()
        )

    def _targets(self) -> Iterator[tuple[str, pd.DataFrame]]:
        targets = pd.read_csv(self._targets_csv)
        targets = (
            targets.drop(targets[targets["parameter"] != self._targets_parameter].index)
            .drop(columns=[targets.columns[0], "scenario", "parameter"])
            .rename(columns={"variable": "Specific"})
            .dropna()
        )
        targets.insert(
            0, "Category", self._parameter_to_category[self._targets_parameter]
        )
        for pattern, replacement in self._parameter_to_category.items():
            # Remove (ignore) if None replacement, else replace
            if replacement is None:
                targets = targets[targets["Specific"] != pattern]
            else:
                targets["Specific"] = targets["Specific"].str.replace(
                    pattern, replacement
                )

        for country, targets_frame in targets.groupby("country"):
            assert isinstance(country, str)
            yield country, targets_frame.drop(columns=["country"])

    def _location_to_path(self, location: str) -> Path:
        return self._location_mapping.get(location, Path("Unknown") / location)

    def __call__(self, output_dir: Path) -> None:
        for file_name, (location, df) in itertools.chain(
            zip(itertools.repeat("techs.csv"), self._intensities()),
            zip(itertools.repeat("targets.csv"), self._targets()),
        ):
            location_dir = output_dir / self._location_to_path(location)
            location_dir.mkdir(exist_ok=True, parents=True)
            df.to_csv(location_dir / file_name, index=False)

        self._indicators().to_csv(output_dir / "indicators.csv", index=False)
