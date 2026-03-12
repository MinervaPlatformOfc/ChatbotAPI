from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    msg: str
    user_name:Optional[str] = None
    session_id:int

