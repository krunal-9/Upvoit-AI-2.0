import time
import os
import re
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query

from models.chatmodel import AIRequest
from core.security import token_required
from services.report_service import process_report,list_report_logs

router = APIRouter(tags=["Reports"])


@router.post("/report")
async def report(
    body: AIRequest,
    user_info: dict = Depends(token_required)
):
    return await process_report(body, user_info)


@router.get("/report-logs")
def get_report_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("createdAt"),
    order: int = Query(-1),
    thread_id: str = Query(None),
    user_info: dict = Depends(token_required),
):
    return list_report_logs(
        page=page,
        limit=limit,
        sort_field=sort,
        sort_order=order,
        thread_id=thread_id,
        user_info=user_info
    )