from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_STRING
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class SimpleTensorBoardWriter:
    def __init__(self, log_dir: Path | str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(self.log_dir))

    def add_scalar(self, tag: str, value: float, step: int = 0) -> None:
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=float(value))])
        self.add_summary(summary, step=step)

    def add_text(self, tag: str, text: str, step: int = 0, description: str | None = None) -> None:
        tensor = TensorProto(
            dtype=DT_STRING,
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
            string_val=[text.encode("utf-8")],
        )
        metadata = SummaryMetadata(
            plugin_data=SummaryMetadata.PluginData(plugin_name="text"),
            summary_description=description or "",
        )
        summary = Summary(value=[Summary.Value(tag=tag, tensor=tensor, metadata=metadata)])
        self.add_summary(summary, step=step)

    def add_png_image(
        self,
        tag: str,
        png_bytes: bytes,
        *,
        height: int,
        width: int,
        step: int = 0,
        colorspace: int = 3,
        description: str | None = None,
    ) -> None:
        summary = Summary(
            value=[
                Summary.Value(
                    tag=tag,
                    image=Summary.Image(
                        height=int(height),
                        width=int(width),
                        colorspace=int(colorspace),
                        encoded_image_string=png_bytes,
                    ),
                    metadata=SummaryMetadata(
                        plugin_data=SummaryMetadata.PluginData(plugin_name="images"),
                        summary_description=description or "",
                    ),
                )
            ]
        )
        self.add_summary(summary, step=step)

    def add_summary(self, summary: Summary, step: int = 0) -> None:
        event = Event(
            wall_time=time.time(),
            step=step,
            summary=summary,
        )
        self._writer.add_event(event)
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "SimpleTensorBoardWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
