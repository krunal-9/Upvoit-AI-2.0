from flask import Flask, request, jsonify,abort
from flask_cors import CORS
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
import logging
from datetime import datetime, timezone, date
from langgraph_sql_agent_chat import run_conversational_query, HumanMessage, AIMessage
from langgraph_sql_agent_chat import run_conversational_query, HumanMessage, AIMessage
from langgraph_sql_agent_reports import run_report_generation
import sqlparse
from sqlparse import tokens as T
from auth import token_required
from pymongo import MongoClient
from bson.objectid import ObjectId
import time
from langgraph_sql_agent_charts import run_chart_generation
from decimal import Decimal
from bson.decimal128 import Decimal128

mongo_uri = os.getenv("MONGO_URI")
mongo_db = os.getenv("MONGO_DB")
mongo_chat_collection = os.getenv("MONGO_CHAT_COLLECTION")
mongo_report_collection = os.getenv("MONGO_REPORT_COLLECTION")
mongo_dashboard_collection = os.getenv("MONGO_DASHBOARD_COLLECTION")
cors_origins = os.getenv("CORS_ORIGINS", "")
cors_origins_list = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]

client = MongoClient(mongo_uri)
db = client[mongo_db]
chat_logs = db[mongo_chat_collection]
report_logs = db[mongo_report_collection]
dashboard_logs = db[mongo_dashboard_collection]


app = Flask(__name__)

CORS(app, resources={
    r"/*": {"origins": cors_origins_list}
})

class ChatMessage(BaseModel):
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: str = ""

class AIRequest(BaseModel):
    message: str
    #max_iterations: Optional[int] = 3
    #chat_history: List[Dict[str, str]] = []
    company_id: Optional[int] = 1  # Default company ID
    thread_id: Optional[str] = None # For session persistence

def format_sql(sql: str) -> str:
    """Format SQL using token-based parsing for clean, readable output."""
    if not sql:
        return ""

    # Parse and format using sqlparse
    formatted = sqlparse.format(
        sql,
        reindent=True,
        keyword_case='upper',
        indent_width=2,
        strip_comments=False,
        use_space_around_operators=True
    )

    return formatted.strip()


@app.route("/api/chat", methods=["POST"])
@token_required
def chat():
    start_time = time.time()   # ⏱ Start timing
    created_at = datetime.now(timezone.utc)
    try:
        try:
            chat_request = AIRequest(**request.get_json())
        except Exception as e:
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
            
        user_info = request.user

        company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
        user_id = str(user_info.get("userId")) if user_info.get("userId") else None

        query = chat_request.message
        thread_id = chat_request.thread_id

        if not any(tag in query.lower() for tag in ['companyid', 'company id']):
            query = f"[CompanyID: {company_id}] {query}"
        
        # Process message
        result = run_conversational_query(query=query, company_id=company_id, thread_id=thread_id)

        # Format the response
        if result.get("is_clarification"):
             # It's a question from the bot
             message = result["natural_response"]
             natural_response = message
        elif result.get("error"):
            # ... (existing error handling)
            message = "Something went wrong. Please try again later."
            natural_response = message
        elif "summary" in result and not result.get("results"):
            # Use the summary for general queries
            message = result["summary"]
            natural_response = message
        elif not result.get("results"):
            message = "ℹ️ No results found for your query."
            natural_response = message
        else:
            # Use the natural response if available, otherwise use the raw results
            natural_response = result.get("natural_response", "Here are your results:")
            if "|" in natural_response and "\n|-" in natural_response:
                message = natural_response
            else:
                message = natural_response + "\n\n" + json.dumps(result["results"], default=str)

        formatted_sql = format_sql(result.get("sql_query", ""))

        response = {
            "message": message,
            "natural_response": natural_response,
            "summary_text": result.get("summary_text", natural_response),
            "error": result.get("error"),
            "company_id": company_id,
            "thread_id": result.get("thread_id"),
            "is_clarification": result.get("is_clarification", False)
        }

        # ⏱ End timing
        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        # 📌 Store log in MongoDB
        log_doc = {
            "companyId": company_id,
            "userId" : user_id,
            "question": chat_request.message,
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
        insert_result =chat_logs.insert_one(log_doc)
        log_id = str(insert_result.inserted_id)
        response["questionId"] = log_id
         # Return response
        return jsonify(content=response)

    except Exception as e:
        try:
            error_log = {
                "companyId": company_id if "company_id" in locals() else None,
                "userId": user_id if "user_id" in locals() else None,
                "question": chat_request.message if "chat_request" in locals() else None,
                "natural_response": None,
                "sqlQuery": None,
                "rating": 0,
                "error": str(e),
                "createdAt": created_at,
                "respondedAt": datetime.now(timezone.utc),
                "executionDuration": None,
                "rawResult": None,
                "version": os.getenv("VERSION"),
                "thread_id": thread_id if "thread_id" in locals() else None,
            }
            insert_result =chat_logs.insert_one(error_log)
            log_id = str(insert_result.inserted_id)
        except:
            pass
        return jsonify({
            "content": {
                "natural_response": None,
                "error": str(e)
            }
        }), 500
    
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "LangChain SQL Chatbot API is running!"})

@app.route("/api/chat-logs", methods=["GET"])
@token_required
def get_chat_logs():
    try:
        # ---- Pagination ----

        user_info = request.user
        companyId = str(user_info.get("companyId")) if user_info.get("companyId") else None
        userId = str(user_info.get("userId")) if user_info.get("userId") else None

        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 20))
        skip = (page - 1) * limit

        # ---- Filters ----
        filter_query = {}

        if companyId:
            filter_query["companyId"] = companyId

        if userId:
            filter_query["userId"] = userId
        
        # ---- Sorting ----
        sort_field = request.args.get("sort", "createdAt")
        sort_order = int(request.args.get("order", -1))  # -1 desc, 1 asc
         
        projection = {
            "_id": 1,                  # will become questionId
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
            chat_logs.find(filter_query,projection)
            .sort(sort_field, sort_order)
            .skip(skip)
            .limit(limit)
        )

        # Convert `_id` to string for JSON
        for r in records:
            r["questionId"] = str(r["_id"])
            del r["_id"]

            # Convert datetime fields to ISO format
            if "createdAt" in r and isinstance(r["createdAt"], datetime):
                r["createdAt"] = r["createdAt"].isoformat() + "Z"

            if "respondedAt" in r and isinstance(r["respondedAt"], datetime):
                r["respondedAt"] = r["respondedAt"].isoformat() + "Z"


        # ---- Response ----
        return jsonify({
            "data": records,
            "pagination": {
                "page": page,
                "limit": limit,
                "totalRecords": total_records,
                "totalPages": (total_records + limit - 1) // limit
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🟦 GET BY ID
@app.route("/api/chat-logs/<id>", methods=["GET"])
@token_required
def get_chat_log_by_id(id):
    try:
        record = chat_logs.find_one({"_id": ObjectId(id)})

        if not record:
            return jsonify({"error": "Record not found"}), 404

        record["_id"] = str(record["_id"])

        return jsonify(record)

    except Exception:
        return jsonify({"error": "Invalid ID"}), 400
    
@app.route("/api/chat-logs/rating", methods=["PUT"])
@token_required
def update_rating():
    try:
        # User info from token
        user_info = request.user
        companyId = str(user_info.get("companyId"))
        userId = str(user_info.get("userId"))

        # Get rating from JSON body
        data = request.get_json()
        new_rating = data.get("rating")
        questionId = data.get("questionId")

        # Validate rating exists
        if new_rating is None:
            return jsonify({"error": "Rating is required"}), 400

        # Validate rating values allowed: 0, 1, 2
        valid_ratings = [0, 1, 2]
        if new_rating not in valid_ratings:
            return jsonify({
                "error": "Invalid rating. Allowed values: 0 (No Response), 1 (Thumbs Up), 2 (Thumbs Down)"
            }), 400

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
            return jsonify({"error": "Record not found"}), 404

        return jsonify({
            "message": "Rating updated successfully",
            "questionId": questionId,
            "rating": new_rating
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/api/charts", methods=["POST"])
@token_required
def charts_endpoint():
    start_time = time.time()   # ⏱ Start timing
    created_at = datetime.now(timezone.utc)
    try:
        try:
            chart_request = AIRequest(**request.get_json())
        except Exception as e:
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
        
        user_info = request.user
        company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
        user_id = str(user_info.get("userId")) if user_info.get("userId") else None

        query = chart_request.message

        if not any(tag in query.lower() for tag in ['companyid', 'company id']):
            query = f"[CompanyID: {company_id}] {query}"

        # Process the message through the chart agent
        result = run_chart_generation(
            query=query, 
            company_id=company_id,
            thread_id=chart_request.thread_id
        )
        
        # Format SQL for display
        formatted_sql = format_sql(result.get("sql_query", ""))
        
        # Create response
        response = {
            "error": result.get("error"),
            "chart_config": result.get("chart_config"),
            "results": result.get("results"),
            "company_id": company_id,
            "thread_id": result.get("thread_id")
        }

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        # 📌 Store log in MongoDB
        log_doc = {
            "companyId": company_id,
            "userId" : user_id,
            "question": chart_request.message,
            "sqlQuery": formatted_sql,
            "error": result.get("error"),
            "createdAt": created_at,
            "respondedAt": datetime.now(timezone.utc),
            "executionDuration": execution_time_ms,
            "rawResult": result,
            "version": os.getenv("VERSION"),
            "thread_id": result.get("thread_id"),
        }
        safe_doc = make_mongo_safe(log_doc)
        insert_result =report_logs.insert_one(safe_doc)
        log_id = str(insert_result.inserted_id)
        response["questionId"] = log_id
         # Return response
        return jsonify(content=response)

    except Exception as e:
        try:
            error_log = {
                "companyId": company_id if "company_id" in locals() else None,
                "userId": user_id if "user_id" in locals() else None,
                "question": chart_request.message if "chart_request" in locals() else None,
                "error": str(e),
                "createdAt": created_at,
                "respondedAt": datetime.now(timezone.utc),
                "executionDuration": 0,
                "rawResult": None,
                "version": os.getenv("VERSION"),
            }
            insert_result =dashboard_logs.insert_one(error_log)
        except:
            pass
        return jsonify({
            "content": {
                "natural_response": None,
                "error": str(e)
            }
        }), 500    


@app.route("/api/report", methods=["POST"])
@token_required
def report():
    start_time = time.time()   # ⏱ Start timing
    created_at = datetime.now(timezone.utc)
    try:
        try:
            report_request = AIRequest(**request.get_json())
        except Exception as e:
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
        
        user_info = request.user
        company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
        user_id = str(user_info.get("userId")) if user_info.get("userId") else None

        query = report_request.message

        if not any(tag in query.lower() for tag in ['companyid', 'company id']):
            query = f"[CompanyID: {company_id}] {query}"

        # Process the message through the report agent
        result = run_report_generation(
            query=query, 
            company_id=company_id,
            thread_id=report_request.thread_id
        )
        
        # Format SQL for display
        formatted_sql = format_sql(result.get("sql_query", ""))
                
        data = []
        columns = []
        execution_error = None

        if result.get("sql_query"):
             # Execution is now handled within run_report_generation if successful
             data = result.get("data", [])
             columns = result.get("columns", [])
             execution_error = result.get("error") if not data and result.get("error") else None
             
             
        # Create response
        response = {
            "error": execution_error or result.get("error"),
            "company_id": company_id,
            "thread_id": result.get("thread_id"),
            "data": data
        }
        
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        print(data)
        # 📌 Store log in MongoDB
        log_doc = {
            "companyId": company_id,
            "userId" : user_id,
            "question": report_request.message,
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
        safe_doc = make_mongo_safe(log_doc)
        insert_result =report_logs.insert_one(safe_doc)
        log_id = str(insert_result.inserted_id)
        response["questionId"] = log_id
        return jsonify(response)
        
    except Exception as e:
        try:
            error_log = {
                "companyId": company_id if "company_id" in locals() else None,
                "userId": user_id if "user_id" in locals() else None,
                "question": report_request.message if "report_request" in locals() else None,
                "error": str(e),
                "createdAt": created_at,
                "respondedAt": datetime.now(timezone.utc),
                "executionDuration": 0,
                "rawResult": None,
                "version": os.getenv("VERSION"),
            }
            insert_result =dashboard_logs.insert_one(error_log)
        except:
            pass
        return jsonify({
            "content": {
                "natural_response": None,
                "error": str(e)
            }
        }), 500    


@app.route("/api/reports/execute", methods=["POST"])
@token_required
def execute_report():
    try:
        try:
            request = ExecuteRequest(**request.get_json())
        except Exception as e:
            return jsonify({"error": "Invalid request", "details": str(e)}), 400
        
        # Validate that the query is a SELECT statement (basic safety)
        if not request.sql.strip().upper().startswith("SELECT") and not request.sql.strip().upper().startswith("WITH"):
            abort(400, description="Only SELECT statements are allowed for execution.")
             
        from langgraph_sql_agent_chat import engine
        from sqlalchemy import text
        import re
        
        with engine.connect() as connection:
            # Extract parameters used in SQL
            params_in_sql = set(re.findall(r'@([a-zA-Z0-9_]+)', request.sql))
            
            # Prepare execution params: default to None if not provided
            execution_params = {param: None for param in params_in_sql}
            
            # Update with provided params
            if request.params:
                execution_params.update(request.params)

            # Use SQLAlchemy text() and bindparams
            stmt = text(request.sql)
            result = connection.execute(stmt, execution_params)
            
            # Serialize rows
            columns = result.keys()
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            
            # Helper to serialize decimals/dates
            serialized_rows = json.loads(json.dumps(rows, default=serialize_data))
            
            response = {
                "data": serialized_rows,
                "columns": list(columns)
}
            return jsonify(content=response)

            
    except Exception as e:
        logging.exception("Error executing report")
        return jsonify({
            "content": {
                "error": str(e)
            }
        }), 500   

def serialize_data(obj):
    """Recursively convert Decimal to float and datetime to str for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_data(i) for i in obj]
    return obj

def make_mongo_safe(obj):
    if isinstance(obj, dict):
        return {k: make_mongo_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_mongo_safe(v) for v in obj]
    if isinstance(obj, Decimal):
        return Decimal128(obj)
    if isinstance(obj, datetime):
        return obj   # pymongo handles datetime automatically
    return obj