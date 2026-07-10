from __future__ import annotations

from dataclasses import dataclass

from .fake_models import FakeFastModel, FakeReasoningModel, FakeWritingModel


@dataclass
class ModelBundle:
    fast_model: FakeFastModel
    reasoning_model: FakeReasoningModel
    writing_model: FakeWritingModel

    @classmethod
    def demo(cls) -> "ModelBundle":
        return cls(FakeFastModel(), FakeReasoningModel(), FakeWritingModel())
