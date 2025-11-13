from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import logging
from datetime import datetime
from langgraph_sql_agent_final import run_conversational_query
import sqlparse
from sqlparse import tokens as T
from auth import token_required

app = Flask(__name__)
CORS(app, resources={
        r"/*": {"origins": [
            "http://192.168.29.16:4200",
            "http://upvoit-ai.appcodzgarage.com",
            "http://192.168.29.16:5080",
            "http://127.0.0.1:5000"
        ]}
    })
class ChatMessage(BaseModel):
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: str = ""

class ChatRequest(BaseModel):
    message: str
    max_iterations: Optional[int] = 3
    chat_history: List[Dict[str, str]] = []


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

@app.post("/api/chat")
@token_required
def chat():
    try:
        try:
            chat_request = ChatRequest(**request.get_json())
        except Exception as e:
            return jsonify({"error": "Invalid request", "details": str(e) }), 400
        
        user_info = request.user  # Retrieved from token
        
        # Add company ID to the query if not present
        query = chat_request.message
        company_id = str(user_info.get("companyId")) if user_info.get("companyId") else None
        
        # Add company ID to query if not already present
        if not any(tag in query.lower() for tag in ['companyid', 'company id']):
            query = f"[CompanyID: {company_id}] {query}"
        
        # Process the message through the agent
        result = run_conversational_query(
            query=query,
            max_iterations=chat_request.max_iterations
        )
        
        # Format the response
        if result.get("error"):
            message = f"❌ Error: {result['error']}"
        elif not result.get("results"):
            message = "ℹ️ No results found for your query."
        else:
            # Return the full results - frontend will handle display
            message = json.dumps(result["results"], default=str)  # Convert any datetime objects to strings
        
        # Format SQL for display
        formatted_sql = format_sql(result.get("sql_query", ""))
        
        # Create response
        response = {
            "message": message,
            "sql": formatted_sql,
            "error": result.get("error"),
            "selected_tables": result.get("selected_tables", []),
            "iteration_count": result.get("iteration_count", 0),
            "company_id": company_id
        }
        
        return jsonify(content=response)
        
    except Exception as e:
        logging.exception("Error in chat endpoint")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

