from functools import cache
from pathlib import Path

import pandas as pd

COUNTRY_MAPPING_CSV = Path(__file__).parent / "country_codes.csv"


@cache
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


def location_to_path(location: str, mapping: dict[str, Path] | None = None) -> Path:
    mapping = mapping if mapping is not None else default_location_mapping()
    return mapping.get(location, Path("Unknown") / location)
