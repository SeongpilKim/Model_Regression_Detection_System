from pydantic import BaseModel, Field
from typing import Literal

class EmailClassification(BaseModel):
    category: Literal["billing", "technical", "account", "general"]
    summary: str = Field(..., description="A single sentence summary under 20 words.")

class PromptConfig(BaseModel):
    email_text: str
    target_model: str = "gpt-4o-mini"