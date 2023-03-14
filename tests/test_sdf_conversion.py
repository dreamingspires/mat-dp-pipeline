import difflib
import logging
import re
from pathlib import Path
from typing import TextIO

import pytest

from mat_dp_pipeline.pipeline.common import ProcessableInput
from mat_dp_pipeline.pipeline.flatten_hierarchy import flatten_hierarchy
from mat_dp_pipeline.pipeline.sparse_to_processable_input import to_processable_input
from mat_dp_pipeline.sdf import standard_data_format as sdf


class DiffError(Exception):
    pass


def to_markdown(output: TextIO, path: Path, year: sdf.Year, inpt: ProcessableInput):
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
    for path, sparse_years_input in flatten_hierarchy(sdf.load(root)):
        for path, year, inpt in to_processable_input(path, sparse_years_input):
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
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid_YearlyFileWithNewTech: Yearly file (2020) introduces new items!"
        ),
    ):
        list(flatten_hierarchy(sdf.load(data_path("Invalid_YearlyFileWithNewTech"))))
