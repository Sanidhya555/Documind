"""Request models shared by the API router and offline component evaluations."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=128)
    # Omitted by the chat UI so adaptive retrieval chooses a score-based context size.
    top_k: int | None = Field(default=None, ge=1, le=20)
    stream: bool = False
    use_cache: bool = True


class SessionRequest(BaseModel):
    session_id: str = Field(..., max_length=128)


class TitleRequest(BaseModel):
    session_id: str = Field(..., max_length=128)
    title: str = Field(..., max_length=120)
