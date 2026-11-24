
from pydantic import BaseModel, Field


class MeaningResult(BaseModel):
    summary: str = Field(min_length=3)
    user_goal: str = Field(min_length=3)
    key_details: list[str]


class StructuredFields(BaseModel):
    problem: str
    duration: str | None = None
    requested_action: str
    important_facts: list[str]
