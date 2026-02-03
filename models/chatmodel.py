from pydantic import BaseModel
from typing import Optional, Any, Dict, List

class AIRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None

class RatingRequest(BaseModel):
    questionId: str
    rating: int

