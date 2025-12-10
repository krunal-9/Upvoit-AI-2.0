import json
import os
import re
import logging
import uuid
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
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
DB_SERVER = os.getenv("DB_SERVER")
DB_DATABASE = os.getenv("DB_DATABASE")
DB_UID = os.getenv("DB_UID")
DB_PWD = os.getenv("DB_PWD")
DB_DRIVER = os.getenv("DB_DRIVER")

odbc_params = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_DATABASE};"
    f"UID={DB_UID};"
    f"PWD={DB_PWD};"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
    "Trusted_Connection=yes;"
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
        ("human", "Query: {query}")
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
       - `labels`: The column name for the X-axis/Category.
       - `values`: The column name for the Y-axis/Value.

    4. **OUTPUT FORMAT:**
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
        ("human", "User Query: {query}\nCompany ID: {company_id}\n\nDetailed Schema for Selected Tables:\n{schema_details}")
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
        
        response = chain.invoke({
            "query": state["query"],
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
        
        response = chain.invoke({
            "query": state["query"],
            "company_id": state["company_id"],
            "schema_details": schema_str
        })
        
        return {
            "sql_query": response["sql"],
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
            
            return {"results": rows}
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

app = workflow.compile()

def run_chart_generation(query: str, company_id: int = 1) -> Dict[str, Any]:
    """Main entry point for chart generation."""
    initial_state = {
        "query": query,
        "company_id": company_id,
        "selected_tables": [],
        "sql_query": "",
        "chart_config": {},
        "results": [],
        "error": None
    }
    
    try:
        final_state = app.invoke(initial_state)
        return {
            "sql_query": final_state.get("sql_query"),
            "chart_config": final_state.get("chart_config"),
            "results": final_state.get("results"),
            "error": final_state.get("error")
        }
    except Exception as e:
        logger.error(f"Error running chart generation: {e}")
        return {"error": str(e)}
