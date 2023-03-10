import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mat_dp_pipeline.common import FileOrPath, validate

Year = int


def validate_tech_units(tech_meta):
    if tech_meta.empty:
        return
    validate(
        (tech_meta.loc[:, "Material Unit"].groupby(level=0).nunique() == 1).all(),
        "There are tech categories with non-unique Material Unit!",
    )
    validate(
        (tech_meta.loc[:, "Production Unit"].groupby(level=0).nunique() == 1).all(),
        "There are tech categories with non-unique Production Unit!",
    )


class InputReader(ABC):
    @property
    @abstractmethod
    def file_pattern(self) -> re.Pattern:
        ...

    @abstractmethod
    def read(self, path: FileOrPath) -> pd.DataFrame:
        ...


class IntensitiesReader(InputReader):
    @property
    def file_pattern(self) -> re.Pattern:
        return re.compile(r"techs_?([0-9]{4})?.csv")

    def read(self, path: FileOrPath) -> pd.DataFrame:
        str_cols = [
            "Category",
            "Specific",
            "Description",
            "Material Unit",
            "Production Unit",
        ]
        return pd.read_csv(
            path,
            index_col=["Category", "Specific"],
            dtype=defaultdict(np.float64, {c: str for c in str_cols}),
            na_values={c: "" for c in str_cols},
        )


class TargetsReader(InputReader):
    @property
    def file_pattern(self) -> re.Pattern:
        return re.compile("targets.csv")

    def read(self, path: FileOrPath) -> pd.DataFrame:
        return pd.read_csv(
            path,
            index_col=["Category", "Specific"],
            dtype=defaultdict(
                np.float64,
                {
                    "Category": str,
                    "Specific": str,
                },
            ),
        )


class IndicatorsReader(InputReader):
    @property
    def file_pattern(self) -> re.Pattern:
        return re.compile(r"indicators_?([0-9]{4})?.csv")

    def read(self, path: FileOrPath) -> pd.DataFrame:
        return pd.read_csv(
            path,
            index_col="Resource",
            dtype=defaultdict(np.float64, {"Resource": str}),
        )


@dataclass(frozen=True, eq=False, order=False)
class StandardDataFormat:
    name: str

    intensities: pd.DataFrame
    intensities_yearly: dict[Year, pd.DataFrame]

    indicators: pd.DataFrame
    indicators_yearly: dict[Year, pd.DataFrame]

    targets: pd.DataFrame | None
    children: dict[str, "StandardDataFormat"]

    tech_meta: pd.DataFrame

    def __post_init__(self):
        self.validate()

    def is_leaf(self) -> bool:
        return self.targets is not None

    def validate(self) -> None:
        def has(item) -> bool:
            return item is not None

        def validate_yearly_keys(base: pd.DataFrame, yearly: dict[Year, pd.DataFrame]):
            base_keys = set(base.index.unique())

            for year, df in yearly.items():
                validate(
                    set(df.index.unique()) <= base_keys,
                    f"{self.name}: Yearly file ({year}) introduces new items!",
                )

        validate(
            has(self.targets) != (len(self.children) > 0),
            f"{self.name}: SDF must either have children in the hierarchy or defined targets (leaf level)",
        )
        validate(
            has(self.intensities) or not has(self.intensities_yearly),
            f"{self.name}: No base intensities, while yearly files provided!",
        )
        validate(
            has(self.indicators) or not has(self.indicators_yearly),
            f"{self.name}: No base indicators, while yearly files provided!",
        )
        validate_yearly_keys(self.intensities, self.intensities_yearly)
        validate_yearly_keys(self.indicators, self.indicators_yearly)

        try:
            validate_tech_units(self.tech_meta)
        except ValueError as e:
            # This isn't a problem just yet - it's possible that the ones with
            # more than one distinct unit won't be in the targets. It won't bother
            # us then. Just warn for now. We'll validate again for the calculation.
            logging.warning(e)

    def _prepare_output_dir(self, root_dir: Path) -> Path:
        output_dir = root_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_intensities(self, root_dir: Path) -> None:
        output_dir = self._prepare_output_dir(root_dir)

        if not self.intensities.empty:
            self.intensities.to_csv(output_dir / "intensities.csv")
        for year, intensities in self.intensities_yearly.items():
            intensities.to_csv(output_dir / f"intensities_{year}.csv")

        for sdf in self.children.values():
            sdf.save_intensities(output_dir)

    def save_indicators(self, root_dir: Path) -> None:
        output_dir = self._prepare_output_dir(root_dir)

        if not self.indicators.empty:
            self.indicators.to_csv(output_dir / "indicators.csv")
        for year, indicators in self.indicators_yearly.items():
            indicators.to_csv(output_dir / f"indicators_{year}.csv")

        for sdf in self.children.values():
            sdf.save_indicators(output_dir)

    def save_targets(self, root_dir: Path) -> None:
        output_dir = self._prepare_output_dir(root_dir)

        if self.targets is not None:
            self.targets.to_csv(output_dir / "targets.csv")
        else:
            for sdf in self.children.values():
                sdf.save_targets(output_dir)

    def save(self, root_dir: Path) -> None:
        self.save_intensities(root_dir)
        self.save_indicators(root_dir)
        self.save_targets(root_dir)


def load(input_dir: Path) -> StandardDataFormat:
    assert input_dir.is_dir()
    targets_reader = TargetsReader()
    intensities_reader = IntensitiesReader()
    indicators_reader = IndicatorsReader()

    def dfs(root: Path) -> StandardDataFormat | None:
        sub_directories = list(filter(lambda p: p.is_dir(), root.iterdir()))

        intensities = None
        intensities_yearly = {}
        indicators = None
        indicators_yearly = {}
        targets = None
        children: dict[str, "StandardDataFormat"] = {}

        files = filter(lambda f: f.is_file(), root.iterdir())
        for file in files:
            if match := intensities_reader.file_pattern.match(file.name):
                year = match.group(1)
                df = intensities_reader.read(file)
                if year is None:
                    # base file
                    intensities = df
                else:
                    year = Year(year)
                    intensities_yearly[year] = df
            elif match := indicators_reader.file_pattern.match(file.name):
                year = match.group(1)
                df = indicators_reader.read(file)
                if year is None:
                    # base file
                    indicators = df
                else:
                    year = Year(year)
                    indicators_yearly[year] = df
            elif targets_reader.file_pattern.match(file.name):
                targets = targets_reader.read(file)

        for sub_directory in sub_directories:
            leaf = dfs(sub_directory)
            if leaf is not None:
                children[sub_directory.name] = leaf

        # If not intensities or indicators were provided, use empty ones
        intensities = pd.DataFrame() if intensities is None else intensities
        indicators = pd.DataFrame() if indicators is None else indicators

        # Ignore leaves with no targets specified
        if targets is None and not sub_directories:
            logging.warning(f"No targets found in {root.name}. Ignoring.")
            return
        else:
            # *Move* metadata from all intensity frames into tech_meta
            tech_meta_cols = ["Description", "Material Unit", "Production Unit"]
            all_intensities = list(intensities_yearly.values()) + [intensities]
            all_meta = [
                i.loc[:, tech_meta_cols] for i in all_intensities if not i.empty
            ]
            if all_meta:
                tech_meta = pd.concat(all_meta).groupby(level=(0, 1)).last()
            else:
                tech_meta = pd.DataFrame()

            for inten in filter(lambda df: not df.empty, all_intensities):
                inten.drop(columns=tech_meta_cols, inplace=True)

            return StandardDataFormat(
                name=root.name,
                intensities=intensities,
                intensities_yearly=intensities_yearly,
                indicators=indicators,
                indicators_yearly=indicators_yearly,
                targets=targets,
                children=children,
                tech_meta=tech_meta,
            )

    root_dfs = dfs(input_dir)
    assert root_dfs is not None
    return root_dfs
