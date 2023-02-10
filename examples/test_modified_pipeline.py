from pathlib import Path
from typing import Iterator

from mat_dp_pipeline import Pipeline
from mat_dp_pipeline.pipeline import LabelledOutput, ProcessableFullType


class ModifiedPipeline(Pipeline):
    def _processable_to_processed(
        self, processable: Iterator[Iterator[ProcessableFullType]]
    ) -> Iterator[LabelledOutput]:
        out = super()._processable_to_processed(processable)
        print(list(out))
        return out


pipeline = ModifiedPipeline()

output = pipeline(Path("./World"))
