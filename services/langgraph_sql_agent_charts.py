import json
import os
import re
import logging
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
def configure_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"chart_agent_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-10s | %(name)-15s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return log_file

configure_logging()
logger = logging.getLogger("chart_agent")

# Database connection
DB_SERVER = os.getenv("DB_SERVER", "tcp:sql5106.site4now.net,1433")
DB_DATABASE = os.getenv("DB_DATABASE", "db_a4a01c_upvoitai")
DB_UID = os.getenv("DB_UID", "db_a4a01c_upvoitai_admin")
DB_PWD = os.getenv("DB_PWD", "upvoit@123")
DB_DRIVER = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

odbc_params = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_DATABASE};"
    f"UID={DB_UID};"
    f"PWD={DB_PWD};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    "Connection Timeout=30;"
)
connection_string = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"
engine = create_engine(connection_string)

# LLM Setup
from utils.llm_config import get_smart_llm
llm = get_smart_llm()

# Schema Cache
class SchemaCache:
    def __init__(self):
        self._cache = {}

    def get(self, file_path: str) -> Any:
        if file_path not in self._cache:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith(".json"):
                        self._cache[file_path] = json.load(f)
                    else:
                        self._cache[file_path] = f.read()
            except Exception as e:
                logger.error(f"Error loading schema {file_path}: {e}")
                return {} if file_path.endswith(".json") else ""
        return self._cache[file_path]

schema_cache = SchemaCache()
DESCRIPTION_A = schema_cache.get("description_A.txt")
DESCRIPTION_B = schema_cache.get("description_B.json")

# State Definition
class ChartState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    company_id: int
    selected_tables: List[str]
    sql_query: str
    chart_config: Dict[str, Any]
    results: List[Dict[str, Any]]
    error: Optional[str]

# --- Prompts ---

def get_table_selection_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """You are a database expert. Your task is to identify the tables relevant to the user's query.
    
    Available Tables:
    {table_summaries}
    
    Return a JSON object with a single key "tables" containing a list of table names.
    Example: {{"tables": ["Invoice", "Clients"]}}
    """),
        MessagesPlaceholder(variable_name="messages"),
    ])

def get_chart_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """You are a Data Visualization Expert. Your goal is to generate a SQL query and a Chart Configuration to visualize the user's request.

    1. **ANALYZE THE REQUEST:**
       - Identify the metric to visualize (e.g., Count of Jobs, Sum of Revenue).
       - Identify the dimension to group by (e.g., Status, Customer, Date).
       - Determine the best chart type:
         - **Pie:** For parts of a whole (e.g., Job Status Distribution).
         - **Bar:** For categorical comparisons (e.g., Jobs per Customer).
         - **Line:** For trends over time (e.g., Jobs per Day).

    2. **GENERATE SQL:**
       - Write a T-SQL query to aggregate the data.
       - **ALWAYS** filter by `CompanyID = {company_id}`.
       - **ALWAYS** filter `IsDeleted = 0` (if applicable).
       - Use `GROUP BY` to aggregate data.
       - Select human-readable labels (e.g., `JobStatus` enum should be mapped if possible, or select the raw value and we'll map it later).
       - **Limit** results if necessary (e.g., Top 10 Customers).
       - **Use the provided schema strictly.** Do not hallucinate column names.

    3. **GENERATE CHART CONFIG:**
       - Create a JSON configuration for the chart.
       - `title`: A clear, descriptive title.
       - `type`: 'pie', 'bar', or 'line'.
       - `labels`: The column name for the X-axis / category (string).
       - `values`: An ARRAY of one or more column names used for Y-axis values.
        - ALWAYS return `values` as an array.
        - Even if there is only ONE metric, wrap it in an array.
         
    4. **REFINEMENT & FOLLOW-UPS:**
       - If the user asks to refine the data (e.g., 'top 10', 'add status'), generate a **NEW, COMPLETE** SQL query.
       - Do NOT try to wrap the previous SQL in a CTE if it risks syntax errors.
       - Ensure specific attention to BALANCED PARENTHESES.
       - If using `WITH`, ensure the `CTE` definition is closed before the final `SELECT`.

    IMPORTANT SCHEMA RULES:
    - `config.values` MUST ALWAYS be a JSON array.
    - NEVER return `values` as a string.
    - Even if there is only ONE metric, wrap it in an array.
    - Single-metric example: "values": ["TotalRevenue"]
    - Multi-metric example: "values": ["TotalRevenue", "TotalProfit"]
         
    5. **OUTPUT FORMAT:**
       Return **ONLY** a valid JSON object.
       - **DO NOT** include any explanations, reasoning, or conversational text.
       - **DO NOT** use markdown formatting (e.g., no ```json or ```sql blocks).
       - The output must be directly parsable by `json.loads()`.
       
       Example:
       {{
           "sql": "SELECT Status, COUNT(*) as Count FROM Jobs WHERE CompanyID = 1 GROUP BY Status",
           "config": {{
               "title": "Job Status Distribution",
               "type": "pie",
               "labels": "Status",
               "values": "Count"
           }}
       }}
    """),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Company ID: {company_id}\n\nDetailed Schema for Selected Tables:\n{schema_details}\n\nUser Query: {query}")
    ])

# --- Helpers ---

def _apply_enum_substitutions(results: List[Dict], table_schemas: Dict) -> List[Dict]:
    """
    Deterministically replace integer enum values with their string representations 
    based on the schema description.
    """
    if not results or not table_schemas:
        return results

    # 1. Build a mapping of {ColumnName: {IntValue: StringValue}}
    enum_mappings = {}
    
    # Iterate through all table definitions in the schema
    for table_name, table_info in table_schemas.items():
        if "columns" not in table_info:
            continue
            
        columns = table_info["columns"]
        # columns can be a list of dicts or a dict of dicts depending on how it was loaded
        cols_iterable = columns if isinstance(columns, list) else columns.values()
        
        for col in cols_iterable:
            col_name = col.get("Column Name")
            description = col.get("Description") or col.get("description", "")
            
            if not col_name or not description:
                continue
                
            # Look for patterns like "1=Created, 2=Scheduled" or "0=No, 1=Yes"
            matches = re.findall(r'(\d+)\s*=\s*([A-Za-z\s]+)(?:,|$)', description)
            
            if matches:
                if col_name not in enum_mappings:
                    enum_mappings[col_name] = {}
                
                for val_str, label in matches:
                    try:
                        val_int = int(val_str)
                        enum_mappings[col_name][val_int] = label.strip()
                    except ValueError:
                        continue

    if not enum_mappings:
        return results

    # 2. Apply substitutions to the results
    processed_results = []
    for row in results:
        new_row = row.copy()
        for col, val in row.items():
            # Check if this column has a mapping
            mapping = None
            for mapped_col in enum_mappings:
                if mapped_col.lower() == col.lower():
                    mapping = enum_mappings[mapped_col]
                    break
            
            if mapping and isinstance(val, int) and val in mapping:
                new_val = mapping[val]
                new_row[col] = new_val
        processed_results.append(new_row)

    return processed_results

# --- Nodes ---

def select_tables(state: ChartState) -> ChartState:
    """Identifies relevant tables using DESCRIPTION_A."""
    try:
        prompt = get_table_selection_prompt()
        chain = prompt | llm | JsonOutputParser()
        
        # Use messages if available, otherwise fallback to query
        messages = state.get("messages", [HumanMessage(content=state["query"])])
        
        response = chain.invoke({
            "messages": messages,
            "table_summaries": DESCRIPTION_A
        })
        
        selected_tables = response.get("tables", [])
        logger.info(f"Selected tables: {selected_tables}")
        
        return {"selected_tables": selected_tables}
    except Exception as e:
        logger.error(f"Error selecting tables: {e}")
        return {"error": str(e)}

def generate_chart_plan(state: ChartState) -> ChartState:
    """Generates SQL and Chart Config using detailed schema."""
    if state.get("error"):
        return state
        
    # Helpers
    def _clean_sql(sql: str) -> str:
        """Cleans the SQL query by removing markdown and formatting issues."""
        if not sql:
            return ""
        
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        sql = sql.strip()
        
        # Remove trailing semicolon
        if sql.endswith(';'):
            sql = sql[:-1]
            
        # Remove outer parentheses if the whole query is wrapped
        if sql.startswith('(') and sql.endswith(')'):
            sql = sql[1:-1].strip()
            
        # Fix specific Common Table Expression (CTE) issue where LLM adds trailing parenthesis
        # Example: WITH ... SELECT ... )
        # A valid query shouldn't end with ) unless it ends with a subquery, but user query was top level.
        # We'll use a heuristic: if it ends with ) but count of ( and ) is not equal.
        if sql.endswith(')') and sql.count('(') < sql.count(')'):
             sql = sql[:-1].strip()
             
        return sql

    try:
        prompt = get_chart_generation_prompt()
        chain = prompt | llm | JsonOutputParser()
        
        # Build detailed schema for selected tables from DESCRIPTION_B
        selected_tables = state.get("selected_tables", [])
        schema_details = []
        
        if isinstance(DESCRIPTION_B, dict):
            for table in selected_tables:
                # Case-insensitive lookup
                table_info = None
                for k, v in DESCRIPTION_B.items():
                    if k.lower() == table.lower():
                        table_info = v
                        break
                
                if table_info:
                    # Format columns for the prompt
                    columns = table_info.get("columns", [])
                    col_desc = []
                    for col in columns:
                        col_name = col.get("Column Name")
                        dtype = col.get("Data Type")
                        desc = col.get("description", "")
                        col_desc.append(f"- {col_name} ({dtype}): {desc}")
                    
                    schema_details.append(f"Table: {table}\n" + "\n".join(col_desc))
        
        schema_str = "\n\n".join(schema_details) if schema_details else "No schema found for selected tables."
        
        messages = state.get("messages", [HumanMessage(content=state["query"])])
        
        response = chain.invoke({
            "messages": messages,
            "query": state["query"], # Still helpful for explicit current intent
            "company_id": state["company_id"],
            "schema_details": schema_str
        })
        
        raw_sql = response.get("sql", "")
        cleaned_sql = _clean_sql(raw_sql)
        
        return {
            "sql_query": cleaned_sql,
            "chart_config": response["config"],
            "error": None
        }
    except Exception as e:
        logger.error(f"Error generating chart plan: {e}")
        return {"error": str(e)}

def execute_chart_query(state: ChartState) -> ChartState:
    """Executes the SQL query."""
    if state.get("error"):
        return state
        
    try:
        with engine.connect() as connection:
            result = connection.execute(text(state["sql_query"]))
            rows = [dict(row._mapping) for row in result]
            
            # Apply Enum Substitutions
            rows = _apply_enum_substitutions(rows, DESCRIPTION_B)
            
            # Success: Append AI Message to history
            msg_content = f"I've generated a **{state['chart_config'].get('type', 'chart')} chart** for *'{state['chart_config'].get('title', 'Query')}'*. See the visualization on the right."
            
            # Embed data for historical restoration
            chart_data = {
                "chart_config": state["chart_config"],
                "execution_results": rows,
                "sql_query": state["sql_query"]
            }
            ai_msg = AIMessage(content=msg_content, additional_kwargs={"data": chart_data})
            
            return {"results": rows, "messages": [ai_msg]}
    except Exception as e:
        logger.error(f"Error executing chart query: {e}")
        return {"error": str(e)}

# --- Graph ---

workflow = StateGraph(ChartState)
workflow.add_node("select_tables", select_tables)
workflow.add_node("planner", generate_chart_plan)
workflow.add_node("executor", execute_chart_query)

workflow.set_entry_point("select_tables")
workflow.add_edge("select_tables", "planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", END)

# Initialize memory
try:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(MONGODB_URI)
    checkpointer = MongoDBSaver(client, db_name="checkpointing_db", collection_name="checkpoints")
except Exception as e:
    logger.warning(f"MongoDB connection failed. Falling back to MemorySaver. Error: {e}")
    checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

async def run_chart_generation(query: str, company_id: int = 1, thread_id: str = None) -> Dict[str, Any]:
    """Main entry point for chart generation."""
    
    # Generate a thread ID if one wasn't provided
    if not thread_id:
        thread_id = f"chart-{uuid.uuid4()}"
        
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if we should resume (add message) or verify new state
    # For simplicity, we always assume new user message comes in "query"
    
    inputs = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "company_id": company_id,
        "selected_tables": [],
        "sql_query": "",
        "chart_config": {},
        "results": [],
        "error": None
    }
    
    try:
        # Use simple invoke. Logic handles appending messages via reducer.
        # Using ainvoke for async execution which leverages run_in_executor for DB ops
        final_state = await app.ainvoke(inputs, config=config)
        
        return {
            "sql_query": final_state.get("sql_query"),
            "chart_config": final_state.get("chart_config"),
            "results": final_state.get("results"),
            "error": final_state.get("error"),
            "thread_id": thread_id
        }
    except Exception as e:
        logger.error(f"Error running chart generation: {e}")
        return {"error": str(e), "thread_id": thread_id}
