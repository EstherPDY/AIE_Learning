from datetime import UTC, datetime
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class Document(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("title")
    @classmethod
    def strip_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be blank")
        return v


class Chunk(BaseModel):
    document_id: str
    index: int = Field(ge=0)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if self.end - self.start != len(self.text):
            raise ValueError("text length must equal end - start")
        return self
