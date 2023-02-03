import difflib
import logging
from pathlib import Path
from typing import TextIO

from pandas.testing import assert_frame_equal
import pytest

from mat_dp_pipeline.sdf_to_input import (
    sdf_to_combined_input,
    sdf_to_processable_input,
    ProcessableInput,
    Year,
)


class DiffError(Exception):
    pass


def to_markdown(output: TextIO, path: Path, year: Year, inpt: ProcessableInput):
    s = f"""
## {path} -- {year}
### Intensities
```
{inpt.intensities.sort_index().to_string()}
```

### Indicators
```
{inpt.indicators.sort_index().to_string()}
```

### Targets
```
{inpt.targets.sort_index().to_string()}
```
"""
    output.write(s)


def process_sdf_to_markdown(root: Path, output: TextIO):
    output.write(f"# {root.name} processed into ProcesableInput\n")
    for path, year, inpt in sdf_to_processable_input(root):
        to_markdown(output, path, year, inpt)


@pytest.mark.parametrize("test_name", ["World", "HierarchyTest", "ScalingTest"])
def test_hierarchy(data_path, test_name: str):
    root = data_path(test_name)
    output_path = Path("/tmp/.mat_dp_pipeline/tests") / (test_name + ".md")
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with output_path.open("w+") as output:
        process_sdf_to_markdown(root, output)
        output.seek(0)
        expected_file = root / "expected.processable_input.md"
        try:
            diff = "".join(
                difflib.unified_diff(
                    expected_file.open("r").readlines(),
                    output.readlines(),
                    fromfile=str(expected_file),
                    tofile=str(output_path),
                )
            )
            if diff:
                raise DiffError("\n" + diff)
        except DiffError as err:
            logging.error(
                f"Output doesn't match. You can find the generated one here: {output_path}\n{err}"
            )
            raise err
        except Exception as err:
            logging.error(
                f"Expected markdown file ({expected_file}) does not exist! You can find the generated one here: {output_path}"
            )
            raise err


def test_failure_new_tech_in_yearly_file(data_path):
    with pytest.raises(AssertionError, match="Yearly file cannot introduce new items!"):
        list(sdf_to_combined_input(data_path("Invalid_YearlyFileWithNewTech")))
