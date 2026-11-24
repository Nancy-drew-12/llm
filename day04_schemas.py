from enum import Enum
from pydantic import BaseModel, Field


class RequestCategory(str, Enum):
    SUPPORT = "support"
    FEEDBACK = "feedback"
    COMPLAINT = "complaint"
    SALES = "sales"
    GENERAL_QUESTION = "general_question"


class IntentClassification(BaseModel):
    category: RequestCategory
    intent: str = Field(min_length=2)