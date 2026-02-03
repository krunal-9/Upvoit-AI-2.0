from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from db.mongo import chat_logs


def update_chat_rating(body, user_info):
    companyId = str(user_info.get("companyId"))
    userId = str(user_info.get("userId"))

    new_rating = body.rating
    questionId = body.questionId

    # Validate rating exists
    if new_rating is None:
        raise HTTPException(
            status_code=400,
            detail="Rating is required"
        )

    # Validate allowed values
    valid_ratings = [0, 1, 2]
    if new_rating not in valid_ratings:
        raise HTTPException(
            status_code=400,
            detail="Invalid rating. Allowed values: 0 (No Response), 1 (Thumbs Up), 2 (Thumbs Down)"
        )

    # Update MongoDB
    update_result = chat_logs.update_one(
        {
            "_id": ObjectId(questionId),
            "companyId": companyId,
            "userId": userId
        },
        {
            "$set": {
                "rating": new_rating,
                "ratingUpdatedAt": datetime.now(timezone.utc)
            }
        }
    )

    if update_result.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Record not found"
        )

    return {
        "message": "Rating updated successfully",
        "questionId": questionId,
        "rating": new_rating
    }

def get_chat_log_by_id(id: str):
    try:
        record = chat_logs.find_one({"_id": ObjectId(id)})

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Record not found"
            )

        record["_id"] = str(record["_id"])
        return record

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid ID"
        )

def list_chat_logs(
    page: int,
    limit: int,
    sort_field: str,
    sort_order: int,
    user_info: dict
):
    companyId = str(user_info.get("companyId")) if user_info.get("companyId") else None
    userId = str(user_info.get("userId")) if user_info.get("userId") else None

    skip = (page - 1) * limit

    # ---- Filters ----
    filter_query = {}

    if companyId:
        filter_query["companyId"] = companyId

    if userId:
        filter_query["userId"] = userId

    # ---- Projection ----
    projection = {
        "_id": 1,
        "question": 1,
        "natural_response": 1,
        "error": 1,
        "respondedAt": 1,
        "createdAt": 1,
        "rating": 1
    }

    # ---- Fetch from Mongo ----
    total_records = chat_logs.count_documents(filter_query)

    records = list(
        chat_logs.find(filter_query, projection)
        .sort(sort_field, sort_order)
        .skip(skip)
        .limit(limit)
    )

    # ---- Transform for JSON ----
    for r in records:
        r["questionId"] = str(r["_id"])
        del r["_id"]

        if "createdAt" in r and isinstance(r["createdAt"], datetime):
            r["createdAt"] = r["createdAt"].isoformat() + "Z"

        if "respondedAt" in r and isinstance(r["respondedAt"], datetime):
            r["respondedAt"] = r["respondedAt"].isoformat() + "Z"

    return {
        "data": records,
        "pagination": {
            "page": page,
            "limit": limit,
            "totalRecords": total_records,
            "totalPages": (total_records + limit - 1) // limit
        }
    }