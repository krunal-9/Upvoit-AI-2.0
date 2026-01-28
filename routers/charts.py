from fastapi import APIRouter, Depends, Query

from models.chatmodel import AIRequest
from core.security import token_required
from services.chart_service import list_chart_logs, process_chart

router = APIRouter(tags=["Charts"])


@router.post("/charts")
async def charts_endpoint(
    body: AIRequest,
    user_info: dict = Depends(token_required)
):
    return await process_chart(body, user_info)


@router.get("/charts-logs")
def get_chat_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("createdAt"),
    order: int = Query(-1),
    thread_id: str = Query(None),
    user_info: dict = Depends(token_required),
):
    return list_chart_logs(
        page=page,
        limit=limit,
        sort_field=sort,
        sort_order=order,
        thread_id=thread_id,
        user_info=user_info
    )