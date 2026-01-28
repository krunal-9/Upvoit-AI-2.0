import time
import json
import os
from datetime import datetime, timezone

from services.langgraph_sql_agent_chat import run_conversational_query
from utils.sql import format_sql
from db.mongo import chat_logs


async def process_chat(body, user_info):
    start_time = time.time()
    created_at = datetime.now(timezone.utc)

    company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
    user_id = str(user_info.get("userId")) if user_info.get("userId") else None

    try:
        query = body.message
        thread_id = body.thread_id

        if not any(tag in query.lower() for tag in ["companyid", "company id"]):
            query = f"[CompanyID: {company_id}] {query}"

        # 🔥 Call existing sync engine
        result  =await run_conversational_query(
            query=query,
            company_id=company_id,
            thread_id=thread_id
        )

        # ---- Response formatting (unchanged) ----
        if result.get("is_clarification"):
            message = result["natural_response"]
            natural_response = message

        elif result.get("error"):
            message = "Something went wrong. Please try again later."
            natural_response = message

        elif "summary" in result and not result.get("results"):
            message = result["summary"]
            natural_response = message

        elif not result.get("results"):
            message = "ℹ️ No results found for your query."
            natural_response = message

        else:
            natural_response = result.get(
                "natural_response", "Here are your results:"
            )
            if "|" in natural_response and "\n|-" in natural_response:
                message = natural_response
            else:
                message = natural_response + "\n\n" + json.dumps(
                    result["results"], default=str
                )

        formatted_sql = format_sql(result.get("sql_query", ""))

        response = {
            "message": message,
            "natural_response": natural_response,
            "summary_text": result.get("summary_text", natural_response),
            "error": result.get("error"),
            "company_id": company_id,
            "thread_id": result.get("thread_id"),
            "is_clarification": result.get("is_clarification", False),
        }

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        log_doc = {
            "companyId": company_id,
            "userId": user_id,
            "question": body.message,
            "natural_response": natural_response,
            "sqlQuery": formatted_sql,
            "rating": 0,
            "error": result.get("error"),
            "createdAt": created_at,
            "respondedAt": datetime.now(timezone.utc),
            "executionDuration": execution_time_ms,
            "rawResult": result,
            "version": os.getenv("VERSION"),
            "thread_id": result.get("thread_id"),
        }

        insert = chat_logs.insert_one(log_doc)
        response["questionId"] = str(insert.inserted_id)

        return {"content": response}

    except Exception as e:
        try:
            error_log = {
                "companyId": company_id,
                "userId": user_id,
                "question": body.message,
                "natural_response": None,
                "sqlQuery": None,
                "rating": 0,
                "error": str(e),
                "createdAt": created_at,
                "respondedAt": datetime.now(timezone.utc),
                "executionDuration": None,
                "rawResult": None,
                "version": os.getenv("VERSION"),
                "thread_id": thread_id,
            }
            chat_logs.insert_one(error_log)
        except Exception:
            pass

        return {
            "content": {
                "natural_response": None,
                "error": str(e),
            }
        }
