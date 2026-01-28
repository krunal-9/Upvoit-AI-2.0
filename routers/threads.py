import re
from fastapi import APIRouter, Depends
from db.mongo import checkpoints, checkpoint_writes
from services.thread_service import get_thread_history
from core.security import token_required

router = APIRouter(tags=["Threads"])

@router.get("/threads")
def get_threads(prefix: str | None = None, 
    user_info: dict = Depends(token_required)):
    query = {}
    if prefix:
        query = {"thread_id": {"$regex": f"^{re.escape(prefix)}"}}
    return {"threads": checkpoints.distinct("thread_id", query)}

@router.delete("/threads/{thread_id}")
def delete_thread(thread_id: str,
    user_info: dict = Depends(token_required)):
    c = checkpoints.delete_many({"thread_id": thread_id})
    w = checkpoint_writes.delete_many({"thread_id": thread_id})
    return {
        "message": f"Thread {thread_id} deleted",
        "deleted_checkpoints": c.deleted_count,
        "deleted_writes": w.deleted_count
    }

@router.get("/threads/{thread_id}/history")
async def thread_history(thread_id: str,
    user_info: dict = Depends(token_required)):
    return await get_thread_history(thread_id)