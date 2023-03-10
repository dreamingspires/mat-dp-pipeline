from pathlib import Path
from typing import IO

FileOrPath = Path | str | IO


def validate(condition: bool, error_message: str | None = None):
    if not condition:
        raise ValueError(error_message)
