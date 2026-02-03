from fastapi import APIRouter, Depends
from fastapi.params import Query
from models.chatmodel import RatingRequest
from core.security import token_required
from services.chat_log_service import update_chat_rating,get_chat_log_by_id,list_chat_logs

router = APIRouter(tags=["Chat Logs"])


@router.put("/chat-logs/rating")
def update_rating(
    body: RatingRequest,
    user_info: dict = Depends(token_required)
):
    return update_chat_rating(body, user_info)

@router.get("/chat-logs/{id}")
def get_chat_log(
    id: str,
    user_info: dict = Depends(token_required)
):
    return get_chat_log_by_id(id)


@router.get("/chat-logs")
def get_chat_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("createdAt"),
    order: int = Query(-1),
    user_info: dict = Depends(token_required),
):
    return list_chat_logs(
        page=page,
        limit=limit,
        sort_field=sort,
        sort_order=order,
        user_info=user_info
    )