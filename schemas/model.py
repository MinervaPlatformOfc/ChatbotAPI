from pydantic import BaseModel
from typing import Optional

class StartRequest(BaseModel):
    API_KEY:str

class ChatRequest(BaseModel):
    msg: str
    user_name:Optional[str]
    session_id:int

class ChatResponse(BaseModel):
    msg: str
    frame: str
    session_id:int