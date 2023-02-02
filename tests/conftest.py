from pathlib import Path

import pandas as pd
import pytest


def _data_path(suffix: Path | None = None) -> Path:
    return Path(__file__).parent / "data" / (suffix or "")


@pytest.fixture()
def world_path() -> Path:
    return _data_path(Path("World"))
