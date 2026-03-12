from fastapi import Depends, FastAPI
from schemas.filter import verify_api_key
from schemas.model import ChatRequest
from agents.frame import start_dialog

client = FastAPI()

@client.post("/")
async def sendMessage(request:ChatRequest, key:str = Depends(verify_api_key)):
    response = start_dialog(
        request.msg,
        request.session_id
    )

    return {
        "msg":response,
        "session_id":request.session_id
    }