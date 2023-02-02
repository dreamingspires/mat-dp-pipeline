from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd

from mat_dp_pipeline.common import FileOrPath


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
            index_col="Material",
            dtype=defaultdict(np.float64, {"Material": str}),
        )
