import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from mat_dp_pipeline.common import FileOrPath

Year = int


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
        def col_filter(c):
            # TODO: add this back in later!!!
            return c not in ("Description", "Material Unit", "Production Unit")

        return pd.read_csv(
            path,
            index_col=["Category", "Specific"],
            usecols=col_filter,
            dtype=defaultdict(
                np.float64,
                {
                    "Category": str,
                    "Specific": str,
                },
            ),
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

    def __post_init__(self):
        self.validate()

    def is_leaf(self) -> bool:
        return self.targets is not None

    def validate(self) -> None:
        if (self.targets is None) == (not self.children):
            raise ValueError(
                "SDF must either have children in the hierarchy or defined targets (leaf level)"
            )

    def save(self, output_dir: Path) -> None:
        assert output_dir.is_dir()
        output_dir = output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.intensities.empty:
            self.intensities.to_csv(output_dir / "intensities.csv")
        for year, intensities in self.intensities_yearly.items():
            intensities.to_csv(output_dir / f"intensities_{year}.csv")

        if not self.indicators.empty:
            self.indicators.to_csv(output_dir / "indicators.csv")
        for year, indicators in self.indicators_yearly.items():
            indicators.to_csv(output_dir / f"indicators_{year}.csv")

        if self.targets is not None:
            self.targets.to_csv(output_dir / "targets.csv")
        else:
            for name, sdf in self.children.items():
                sdf.save(output_dir / name)


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
        if intensities is None:
            assert (
                not intensities_yearly
            ), "No base intensities, while yearly files provided!"
            intensities = pd.DataFrame()
        if indicators is None:
            assert not indicators, "No base indicators, while yearly files provided!"
            indicators = pd.DataFrame()

        # Ignore leaves with no targets specified
        if targets is None and not sub_directories:
            logging.warning(f"No targets found in {root.name}. Ignoring.")
            return
        else:
            return StandardDataFormat(
                name=root.name,
                intensities=intensities,
                intensities_yearly=intensities_yearly,
                indicators=indicators,
                indicators_yearly=indicators_yearly,
                targets=targets,
                children=children,
            )

    root_dfs = dfs(input_dir)
    assert root_dfs is not None
    return root_dfs
