import time
import os
from datetime import datetime, timezone

from fastapi import HTTPException

from services.langgraph_sql_agent_reports import run_report_generation
from utils.sql import format_sql
from utils.mongo_utils import make_mongo_safe, serialize_mongo_value
from db.mongo import report_logs, dashboard_logs


async def process_report(body, user_info):
    start_time = time.time()
    created_at = datetime.now(timezone.utc)

    company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
    user_id = str(user_info.get("userId")) if user_info.get("userId") else None

    try:
        query = body.message

        if not any(tag in query.lower() for tag in ["companyid", "company id"]):
            query = f"[CompanyID: {company_id}] {query}"

        # 🔥 Call your existing async engine
        result = await run_report_generation(
            query=query,
            company_id=company_id,
            thread_id=body.thread_id
        )

        formatted_sql = format_sql(result.get("sql_query", ""))

        data = []
        columns = []
        execution_error = None

        if result.get("sql_query"):
            data = result.get("data", [])
            columns = result.get("columns", [])
            execution_error = result.get("error") if not data and result.get("error") else None

        response = {
            "error": execution_error or result.get("error"),
            "company_id": company_id,
            "thread_id": result.get("thread_id"),
            "data": data,
        }

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        log_doc = {
            "companyId": company_id,
            "userId": user_id,
            "question": body.message,
            "sqlQuery": formatted_sql,
            "data": data,
            "columns": columns,
            "scratchpad": result.get("scratchpad"),
            "error": result.get("error"),
            "createdAt": created_at,
            "respondedAt": datetime.now(timezone.utc),
            "executionDuration": execution_time_ms,
            "rawResult": result,
            "version": os.getenv("VERSION"),
            "thread_id": result.get("thread_id"),
        }

        insert = report_logs.insert_one(make_mongo_safe(log_doc))
        response["questionId"] = str(insert.inserted_id)

        return response

    except Exception as e:
        try:
            error_log = {
                "companyId": company_id,
                "userId": user_id,
                "question": body.message,
                "error": str(e),
                "createdAt": created_at,
                "respondedAt": datetime.now(timezone.utc),
                "executionDuration": 0,
                "rawResult": None,
                "version": os.getenv("VERSION"),
            }
            dashboard_logs.insert_one(error_log)
        except Exception:
            pass

        raise HTTPException(
            status_code=500,
            detail={
                "natural_response": None,
                "error": str(e)
            }
        )


def list_report_logs(
    page: int,
    limit: int,
    sort_field: str,
    sort_order: int,
    thread_id: str,
    user_info: dict
):
    companyId = str(user_info.get("companyId")) if user_info.get("companyId") else None    
    userId = str(user_info.get("userId")) if user_info.get("userId") else None

    skip = (page - 1) * limit
    # ---- Filters ----
    filter_query = {}    

    if companyId:
        filter_query["companyId"] = companyId
    # if userId:
    #     filter_query["userId"] = userId   
    if thread_id:
        filter_query["thread_id"] = thread_id
    # ---- Projection ----    
    projection = {
        "_id": 1,
        "question": 1,
        "data": 1,
        "error": 1,
        "respondedAt": 1,
        "createdAt": 1,
        "thread_id": 1
    }

    # ---- Fetch from Mongo ----
    total_records = report_logs.count_documents(filter_query)
    records = list(
        report_logs.find(filter_query, projection)
        .sort(sort_field, sort_order)
        .skip(skip)
        .limit(limit)
    )
    safe_records = []

    # ---- Transform for JSON ----
    for r in records:
        r["questionId"] = str(r["_id"])
        del r["_id"]
        safe_record = serialize_mongo_value(r)
        safe_records.append(safe_record)
        if "createdAt" in r and isinstance(r["createdAt"], datetime):
            r["createdAt"] = r["createdAt"].isoformat() + "Z"

        if "respondedAt" in r and isinstance(r["respondedAt"], datetime):
            r["respondedAt"] = r["respondedAt"].isoformat() + "Z"

    return {
        "data": safe_records,
        "pagination": {
            "page": page,
            "limit": limit,
            "totalRecords": total_records,
            "totalPages": (total_records + limit - 1) // limit
        }
    }