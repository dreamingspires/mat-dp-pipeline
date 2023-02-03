from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture()
def data_path():
    def inner(suffix: Path | None = None) -> Path:
        p = Path(__file__).parent / "data" / (suffix or "")
        print(p)
        return Path(__file__).parent / "data" / (suffix or "")

    return inner
