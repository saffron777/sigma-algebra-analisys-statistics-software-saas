"""Base class for request models: tolerant of blank form fields."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

Confidence = Annotated[float, Field(gt=0, lt=1)]
Probability = Annotated[float, Field(gt=0, lt=1)]
NonEmpty = Annotated[str, Field(min_length=1)]


class RequestModel(BaseModel):
    """Unknown keys are ignored and ``""`` becomes ``None`` for optional fields (empty HTML inputs)."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    blank_as_none: ClassVar[bool] = True  # models where "" is meaningful (e.g. clearing a note) set this to False

    @model_validator(mode="before")
    @classmethod
    def _blank_to_none(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not cls.blank_as_none:
            return data
        cleaned = dict(data)
        for name, value in data.items():
            field = cls.model_fields.get(name)
            if value == "" and field is not None and not field.is_required() and field.default is None:
                cleaned[name] = None
        return cleaned
