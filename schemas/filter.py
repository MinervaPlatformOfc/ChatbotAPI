import os
from dotenv import load_dotenv
from fastapi import HTTPException,Security
from fastapi.security.api_key import APIKeyHeader

load_dotenv()

API_KEY = os.environ.get("API_KEY")
API_KEY_NAME = os.environ.get("API_KEY_NAME")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Chave de API inválida")
    return api_key
