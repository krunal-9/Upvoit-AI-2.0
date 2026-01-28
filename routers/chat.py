from fastapi import APIRouter, Depends

from models.chatmodel import AIRequest
from core.security import token_required
from services.chat_service import process_chat

router = APIRouter(tags=["Chat"])


@router.post("/chat")
async def chat(
    body: AIRequest,
    user_info: dict = Depends(token_required)
):
    return await process_chat(body, user_info)
