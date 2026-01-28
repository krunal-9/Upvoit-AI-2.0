import logging
from datetime import datetime

from langchain_core.messages import HumanMessage

from services.langgraph_sql_agent_chat import app as chat_agent_app
from services.langgraph_sql_agent_charts import app as chart_agent_app
from services.langgraph_sql_agent_reports import app as report_agent_app


async def get_thread_history(thread_id: str):
    """
    Get the history and state of a thread.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}

        if thread_id.startswith("chart-"):
            state_snapshot = await chart_agent_app.aget_state(config)
        elif thread_id.startswith("report-"):
            state_snapshot = await report_agent_app.aget_state(config)
        else:
            state_snapshot = await chat_agent_app.aget_state(config)

        if not state_snapshot.values:
            return {"history": [], "state": {}}

        # ---- Extract Messages ----
        messages_raw = state_snapshot.values.get("messages", [])
        history = []

        for msg in messages_raw:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content

            msg_data = getattr(msg, "additional_kwargs", {}).get("data")

            history.append({
                "role": role,
                "content": str(content),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": msg_data
            })

        # ---- Extract Remaining State ----
        state_values = {
            k: v for k, v in state_snapshot.values.items()
            if k != "messages"
        }

        return {
            "history": history,
            "state": state_values
        }

    except Exception as e:
        logging.exception(f"Error getting history for thread {thread_id}")
        return {
            "history": [],
            "state": {}
        }
