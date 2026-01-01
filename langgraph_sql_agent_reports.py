import json
import os
import re
import logging
import uuid
import time
from functools import lru_cache
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Logging configuration
class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"

    format_str = "%(asctime)s | %(levelname)-10s | %(name)-15s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def configure_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"report_agent_{timestamp}.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-10s | %(name)-15s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    return log_file

# Initialize logging
configure_logging()
logger = logging.getLogger(__name__)

def _format_data(data: Any, indent: int = 2) -> str:
    if isinstance(data, str):
        return f'"{data}"'
    elif isinstance(data, (int, float, bool)) or data is None:
        return str(data)
    elif isinstance(data, (list, tuple)):
        return "[" + ", ".join(_format_data(item) for item in data) + "]"
    elif isinstance(data, dict):
        return (
            "{" + ", ".join(f'"{k}": {_format_data(v)}' for k, v in data.items()) + "}"
        )
    else:
        return str(data)

def log_agent_step(
    agent_name: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO",
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    log_msg = f"{agent_name}: {message}"
    if data is not None and len(data) > 0:
        try:
            formatted_data = "\n".join(
                f"{k}: {_format_data(v)}" for k, v in data.items()
            )
            log_msg = f"{log_msg}\n{formatted_data}"
        except Exception as e:
            logger.warning(f"Failed to format log data: {e}")
    logger.log(log_level, log_msg)

def log_error(
    agent_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR",
) -> None:
    error_data = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        **({} if context is None else context),
    }
    log_agent_step(agent_name, "Error occurred", error_data, level)

load_dotenv()

# Schema Cache
class SchemaCache:
    def __init__(self):
        self._cache = {}
        self._last_modified = {}

    def get(self, file_path: str) -> Any:
        try:
            current_mtime = os.path.getmtime(file_path)
            if (
                file_path not in self._cache
                or file_path not in self._last_modified
                or self._last_modified[file_path] < current_mtime
            ):
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith(".json"):
                        self._cache[file_path] = json.load(f)
                    else:
                        self._cache[file_path] = f.read()
                self._last_modified[file_path] = current_mtime
            return self._cache[file_path]
        except Exception as e:
            log_error(f"SchemaCache", e, {"file_path": file_path})
            raise

    def invalidate(self, file_path: str = None):
        if file_path:
            self._cache.pop(file_path, None)
            self._last_modified.pop(file_path, None)
        else:
            self._cache.clear()
            self._last_modified.clear()

schema_cache = SchemaCache()

def load_description_a() -> str:
    return schema_cache.get("description_A.txt")

def load_description_b() -> dict:
    return schema_cache.get("description_B.json")

try:
    DESCRIPTION_A = load_description_a()
    DESCRIPTION_B = load_description_b()
except Exception as e:
    log_error("SchemaLoader", e, {"files": ["description_A.txt", "description_B.json"]})
    raise

# LLM Configuration
from utils.llm_config import get_fast_llm, get_smart_llm

try:
    fast_llm = get_fast_llm()
    smart_llm = get_smart_llm()
except Exception as e:
    print(f"Error initializing language models: {str(e)}")
    raise

# DB Connection
try:
    driver = "ODBC Driver 18 for SQL Server"
    DB_SERVER = os.getenv("DB_SERVER", "tcp:sql5106.site4now.net,1433")
    DB_DATABASE = os.getenv("DB_DATABASE", "db_a4a01c_upvoitai")
    DB_UID = os.getenv("DB_UID", "db_a4a01c_upvoitai_admin")
    DB_PWD = os.getenv("DB_PWD", "upvoit@123")

    odbc_params = (
        f"DRIVER={{{driver}}};"
        f"SERVER={DB_SERVER};"
        f"DATABASE={DB_DATABASE};"
        f"UID={DB_UID};"
        f"PWD={DB_PWD};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
        "Connection Timeout=15;"
        "Login Timeout=15;"
    )
    DATABASE_URI = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"
    engine = create_engine(DATABASE_URI, pool_pre_ping=True)
except Exception as e:
    print(f"Failed to establish database connection. Error: {str(e)}")
    raise

def validate_sql_syntax(sql: str) -> Tuple[bool, Optional[str]]:
    """Basic SQL syntax validation to catch common errors before execution."""
    errors = []
    
    # Check for unbalanced parentheses, ignoring those in strings and comments
    def check_balanced_parentheses(s):
        stack = []
        in_string = False
        string_char = None
        in_comment = False # -- comment
        in_block_comment = False # /* comment */
        i = 0
        while i < len(s):
            char = s[i]
            
            # Handle comments
            if not in_string and not in_comment and not in_block_comment:
                if s[i:i+2] == '--':
                    in_comment = True
                    i += 2
                    continue
                elif s[i:i+2] == '/*':
                    in_block_comment = True
                    i += 2
                    continue
            
            if in_comment:
                if char == '\n':
                    in_comment = False
                i += 1
                continue
                
            if in_block_comment:
                if s[i:i+2] == '*/':
                    in_block_comment = False
                    i += 2
                else:
                    i += 1
                continue
            
            # Handle strings
            if char == "'" or char == '"':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    # Check for escaped quote (e.g. 'It''s')
                    if i + 1 < len(s) and s[i+1] == string_char:
                        i += 1 # Skip next char
                    else:
                        in_string = False
            
            # Handle parentheses
            if not in_string:
                if char == '(':
                    stack.append(i)
                elif char == ')':
                    if not stack:
                        return False, f"Extra closing parenthesis at position {i}"
                    stack.pop()
            
            i += 1
            
        if stack:
            return False, f"Unclosed parenthesis starting at position {stack[-1]}"
        return True, None

    is_balanced, paren_error = check_balanced_parentheses(sql)
    if not is_balanced:
        errors.append(paren_error)
    
    # Check for unbalanced quotes (simple check, might be redundant with above but good for specific error msg)
    single_quotes = sql.count("'") - sql.count("\\'")
    if single_quotes % 2 != 0:
        # Double check with the smarter logic if needed, but simple count is usually a good heuristic for simple SQL
        # However, T-SQL escapes single quotes with another single quote, not backslash.
        # So 'It''s' has 4 quotes (even). 'It's' has 3 (odd).
        # This simple check is actually correct for T-SQL standard escaping.
        errors.append("Unbalanced single quotes")
    
    # Check for common syntax errors
    common_errors = [
        (r',\s*\)', "Comma before closing parenthesis"),
        (r'\(\s*,', "Comma right after opening parenthesis"),
        (r'SELECT\s*,', "SELECT with comma but no column"),
        (r',\s*,', "Double comma"),
        (r'FROM\s*,', "FROM with comma"),
        (r'WHERE\s*,', "WHERE with comma"),
        (r'AND\s*$', "AND at end of query"),
        (r'OR\s*$', "OR at end of query"),
        (r'BETWEEN\s+[^ ]+\s+AND\s*$', "INCOMPLETE BETWEEN clause"),
        (r'^\s*TOP\s+', "TOP without SELECT - missing SELECT keyword"),
    ]
    
    for pattern, description in common_errors:
        if re.search(pattern, sql, re.IGNORECASE | re.MULTILINE):
            errors.append(description)
    
    # Check for DECLARE without usage
    # declare_vars = re.findall(r'DEclare\s+@(\w+)', sql, re.IGNORECASE)
    
    # Check for variable usage without declaration - SKIPPED for templates
    # used_vars = re.findall(r'@(\w+)', sql)
    # declared_vars = set(v.upper() for v in declare_vars)
    
    # Common system variables to ignore
    # system_vars = {'IDENTITY', 'ROWCOUNT', 'ERROR', 'TRANCOUNT', 'VERSION', 'SPID', 'FETCH_STATUS'}
    
    # for var in used_vars:
    #     if var.upper() in system_vars or var.upper().startswith('@@'):
    #         continue
            
    #     if var.upper() not in declared_vars:
    #         # errors.append(f"Variable @{var} is used but not declared")
    pass
    #         errors.append(f"Variable @{var} is used but not declared")
    
    # Check for incomplete statements
    if sql.strip().endswith('AND') or sql.strip().endswith('OR'):
        errors.append("Query ends with AND or OR (incomplete condition)")
    
    # Check if DECLARE is followed by a valid SQL statement
    if 'DECLARE' in sql.upper():
        lines = sql.split('\n')
        declare_found = False
        for i, line in enumerate(lines):
            if 'DECLARE' in line.upper():
                declare_found = True
            elif declare_found and line.strip():
                # Check if the next non-empty line after DECLARE starts with a SQL keyword
                # We want to allow SET, IF, etc. too if needed
                if not re.match(r'^(WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|SET|IF|WHILE|BEGIN|PRINT|EXEC|EXECUTE)\s', line.strip(), re.IGNORECASE):
                    # Check if it might be a continuation of DECLARE or a missing SELECT
                    # If it starts with @, it's likely another declare or set
                    if not line.strip().startswith('@') and re.match(r'^[A-Z_]+$', line.strip(), re.IGNORECASE):
                         # Only flag if it looks like a keyword that should be start of statement
                         if line.strip().upper() in ['FROM', 'WHERE', 'GROUP', 'ORDER', 'HAVING']:
                             errors.append(f"Possible missing SELECT after DECLARE statements (line {i+1}: {line.strip()})")
                break
    
    # Check for missing table aliases in JOIN conditions
    join_pattern = r'(INNER|LEFT|RIGHT|FULL)\s+JOIN\s+(\w+)\s+(?:AS\s+)?(\w+)?\s+ON'
    joins = re.findall(join_pattern, sql, re.IGNORECASE)
    for join_type, table, alias in joins:
        if alias:
            # Check if alias is used in the ON condition
            on_pattern = rf'JOIN\s+{re.escape(table)}\s+(?:AS\s+)?{re.escape(alias)}\s+ON\s+(.*?)(?:\s+(?:INNER|LEFT|RIGHT|FULL|JOIN|WHERE|ORDER|GROUP|HAVING|$))'
            on_match = re.search(on_pattern, sql, re.IGNORECASE | re.DOTALL)
            if on_match:
                on_condition = on_match.group(1)
                if alias.lower() not in on_condition.lower() and table.lower() not in on_condition.lower():
                    # This is also a bit aggressive but we'll keep it for now
                    pass
    
    return len(errors) == 0, "\n".join(errors) if errors else None

def extract_sql(llm_output: str) -> Tuple[str, Optional[str]]:
    # Step 1: Extract and remove scratchpad to prevent false positives in tag search
    # (e.g. "Return code in <sql> tags")
    scratchpad_match = re.search(
        r"<scratchpad>(.*?)</scratchpad>", llm_output, re.DOTALL | re.IGNORECASE
    )
    scratchpad_text = scratchpad_match.group(1).strip() if scratchpad_match else None
    
    # Remove scratchpad from the text we process for SQL
    if scratchpad_match:
        # We replace the whole match with empty string
        clean_output = llm_output.replace(scratchpad_match.group(0), "").strip()
    else:
        clean_output = llm_output.strip()
            
    sql = ""
    
    # Method 1: <sql> tags (Case Insensitive)
    lower_output = clean_output.lower()
    start_tag = "<sql>"
    end_tag = "</sql>"
    
    start_idx = lower_output.find(start_tag)
    if start_idx != -1:
        # Found opening tag
        content_start = start_idx + len(start_tag)
        
        end_idx = lower_output.find(end_tag, content_start)
        if end_idx != -1:
            # Found closing tag
            sql = clean_output[content_start:end_idx]
        else:
            # No closing tag, take rest of string
            sql = clean_output[content_start:]
    
    else:
        # Method 2: Markdown block
        code_blocks = re.findall(r"```(?:sql)?\s*(.*?)\s*```", clean_output, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            sql = "\n".join(block.strip() for block in code_blocks)
        
        else:
            # Method 3: Fallback - Raw output
            # Since we already stripped scratchpad, clean_output is just the SQL + potentially conversational text
            sql = clean_output
            
            # If the user output "tags.\n\nSELECT...", we still have "tags."
            # We heuristic: find start of SELECT / WITH / INSERT
            match = re.search(r"(WITH\s|SELECT\s|INSERT\s|UPDATE\s|DELETE\s|DECLARE\s)", sql, re.IGNORECASE)
            if match:
                sql = sql[match.start():]

    # Final Cleanup
    # Remove any remaining XML tags that might have leaked (e.g. closing tags if malformed)
    sql = re.sub(r"<[^>]+>", "", sql).strip()
    
    # Remove standard comments
    sql = re.sub(r"^\s*--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    
    # Ensure it ends with semicolon
    if sql and not sql.endswith(";"):
        sql += ";"
    
    return sql, scratchpad_text

def get_time() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def get_report_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a SQL expert specializing in generating REUSABLE REPORT TEMPLATES.
Your task is to create a parameterized T-SQL query that serves as a flexible report for the user.

Follow these steps to generate the best possible query:

1. **UNDERSTAND THE GOAL & CONTEXT:**
   - **LATEST REQUEST IS KING:** Focus on the *latest* user request (e.g., "Add modified date").
   - **CONTEXTUAL MODIFICATION:** If the user is asking to *modify* an existing report (e.g., "add a column", "filter by X"), you **MUST PRESERVE** the existing query's structure (columns, joins, logic) and ONLY apply the requested changes.
     - **DO NOT** remove existing columns unless explicitly asked.
     - **DO NOT** change column aliases unless explicitly asked.
   - **NEW REPORT:** If the user asks for a completely different report, start fresh.

2. **ANALYZE THE SCHEMA:**
   - Review the available tables and their columns.
   - Identify which tables contain the required data.
   - Note any relationships between tables (foreign keys).

3. **COLUMN SELECTION PRINCIPLES (CRITICAL):**
   - **Distinguish Keys:**
     - **Surrogate Keys:** (e.g., GUIDs, auto-increments, IDs) are for system use. DO NOT include them unless the user explicitly asks for an ID.
     - **Natural Keys:** (e.g., Names, Codes, Titles) are for humans. ALWAYS prefer these over surrogate keys.
   - **Audit Data:** Columns like `CreatedBy`, `ModifiedDate`, `ConcurrencyStamp` are metadata. Try not to include them.
   - **User Intent:**
     - If the user asks for "Jobs", they want the *Job Name*, *Date*, and *Status*. They do NOT want the `JobId` or `CompanyId`.
     - If the user asks for "Clients", they want the *Customer Name*, not the `CustomerId`.
   - **Readability:**
     - Prefer descriptive text columns over numeric codes.
     - Limit output to the most relevant 6-8 columns.

4. **FILTERING LOGIC:**
   - **Date Range:** You MUST filter the primary date column using T-SQL functions (like `GETDATE()`). Do NOT use `@FromDate` or `@ToDate` unless explicitly asked.
   - **NO DYNAMIC FILTERS:** Do NOT add optional filters for every column. Only add filters that are explicitly requested by the user.

5. **QUERY STRUCTURE:**
   ```sql
   SELECT
       t.Column1 AS [Friendly Name 1],
       t.Column2 AS [Friendly Name 2],
       ...
   FROM Table t
   ...
   WHERE
       t.CompanyId = 1
       AND (t.IsDeleted = 0 OR t.IsDeleted IS NULL)
       -- DATE FILTER (Self-contained)
       AND t.DateColumn BETWEEN DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0) AND EOMONTH(GETDATE())
   ```

6. **CRITICAL RULES:**
   - **DO NOT DECLARE VARIABLES:** The frontend/API will handle declarations. Your output should START with `SELECT`.
   - **NO PARAMETERS:** You must NOT use any `@Params` (like `@FromDate`, `@Status`, `@JobNumber`). All logic must be hardcoded or self-contained in the SQL.
     - **Instead of `@FromDate`**, use: `DATEADD(month, DATEDIFF(month, 0, GETDATE()), 0)` (Start of current month).
     - **Instead of `@ToDate`**, use: `EOMONTH(GETDATE())` (End of current month).
   - **Multi-Tenancy:** ALWAYS filter by `CompanyId = {company_id}`.
   - **Soft Deletes:** ALWAYS filter `IsDeleted = 0`.
   - **Recurring Jobs (CRITICAL):** When querying recurring jobs (e.g., Visits, Schedules), the 'Start Time' and 'End Time' in the database might be the *series* start/end. You MUST construct the actual instance datetime using the `VisitDate` (or equivalent) combined with the time component of the start/end columns.
     - Example: `DATEADD(day, DATEDIFF(day, '19000101', VisitDate), CAST(StartTime AS DATETIME))`
     - Use this computed date for the SELECT clause AND the `@FromDate/@ToDate` filter.
   - **No Technical Columns:** ABSOLUTELY NO technical/ID columns in SELECT unless explicitly requested.
   - **ENUM/LOOKUP HANDLING (MANDATORY):**
     - **CHECK THE SCHEMA:** For every column you select, check its `description` in the provided schema.
     - **DYNAMIC TRANSLATION:** If the description defines a mapping (e.g., "1=Created, 2=Scheduled"), you **MUST** generate a `CASE` statement to translate the integer values into their string representations.
     - **NO RAW INTEGERS:** Never output the raw integer for a verified Enum column.
     - **Example:**
       If schema says `Example: 1=Hello, 2=World`:
       ```sql
       CASE [Example]
           WHEN 1 THEN 'Hello'
           WHEN 2 THEN 'World'
           ELSE CAST([Example] AS VARCHAR)
       END AS [Example]
       ```

7. **COMMON PITFALLS TO AVOID:**
   - Don't compare string IDs with integer IDs.
   - Don't use string concatenation for multiple IDs.
   - Ensure data types match in WHERE conditions and JOINs.

8. **THINK STEP BY STEP (GUIDED THINKING):**
   - **Step 1: Analyze Request Type:** Is this a new report or a modification?
   - **Step 2: Review Previous Columns and query:**
     - Previous Columns: {previous_columns}
     - *Decision:* If modifying, start with these columns.
   - **Step 3: Modify Columns (CRITICAL):**
     - **START** with the complete list of Previous Columns.
     - **ADD** requested columns (e.g., "Modified Date").
     - **REMOVE** columns **ONLY** if the user explicitly says "remove X" or "drop X".
     - **FORBIDDEN:** Do NOT add or drop any column silently. If in doubt, keep it.
     - Translate Enums? (Check schema).
   - **Step 4: Filters & Logic:**
     - Apply date filters (Current Month default or requested range).
     - Apply CompanyId/IsDeleted.
     - Apply specific filters requested.
   - **Step 5: Finalize:**
     - Which tables contain this information?
     - How should these tables be joined?
     - For each selected column, have I added the corresponding dynamic filter?
     - Review the final query.
     - Ensure it matches the user's intent.
     - Ensure it's valid SQL.

9. **OUTPUT FORMAT:**
   - Return ONLY the SQL query inside <sql> tags.
   - Include a <scratchpad> for your thinking.

DATABASE SCHEMA:
{schema}
Error history (if any):
{error_history}
""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

class AgentState(TypedDict):
    run_id: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    company_id: int
    selected_tables: List[str]
    table_schemas: Dict[str, Any]
    generated_query: str
    error: str
    iteration_count: int
    max_iterations: int
    description_a: str
    last_scratchpad: Optional[str]
    error_history: List[str]
    execution_results: List[Dict[str, Any]]
    columns: List[str]

# Node 1: Table Selector (Same as Chat Agent)
def table_selector(state: AgentState) -> AgentState:
    try:
        log_agent_step("TableSelector", "Starting table selection", {"user_query": state["user_query"]})
        description_a = state.get("description_a", DESCRIPTION_A)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a database expert. Select relevant tables for the user's report request.
    AVAILABLE TABLES:
    {table_descriptions}
    USER QUERY: {user_query}
    INSTRUCTIONS:
    1. Select tables necessary for the report.
    2. Return JSON: {{"tables": ["Table1", "Table2"]}}
    """,
                )
            ]
        )
        chain = prompt | fast_llm | JsonOutputParser()
        response = chain.invoke({"table_descriptions": description_a, "user_query": state["user_query"]})
        selected_tables = response.get("tables", [])
        log_agent_step("TableSelector", "Selected tables", {"selected_tables": selected_tables})
        return {**state, "selected_tables": selected_tables}
    except Exception as e:
        log_error("TableSelector", e)
        return {**state, "error": str(e)}

def load_table_schema(table: str, description_b: Dict) -> Optional[Dict]:
    exact = next((t for t in description_b if t.lower() == table.lower()), None)
    if exact and "columns" in (info := description_b[exact]):
        columns_data = info["columns"]
        if isinstance(columns_data, list):
            # Create a dict mapping Column Name -> Column Details
            columns_dict = {}
            for col in columns_data:
                if isinstance(col, dict) and "Column Name" in col:
                    col_name = col["Column Name"]
                    # Store the full column details, including Description, Data Type, etc.
                    columns_dict[col_name] = col
            
            return {
                "description": info.get("description", ""),
                "columns": columns_dict,
            }
    return None
    return None

# Node 2: Report Query Generator
def query_generator(state: AgentState) -> AgentState:
    run_id = state.get("run_id", str(uuid.uuid4()))
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return {**state, "error": "Max iterations reached"}

    selected_tables = state.get("selected_tables", [])
    table_schemas = {}
    schema_table_mapping = {}
    
    available_tables = list(DESCRIPTION_B.keys())
    for table in selected_tables:
        schema = load_table_schema(table, DESCRIPTION_B)
        if schema:
            # Find exact match in available tables
            exact_match = next(
                (t for t in available_tables if t.lower() == table.lower()),
                None,
            )
            table_schemas[exact_match] = schema
            schema_table_mapping[table] = exact_match

    # Manage error history
    previous_error = state.get("error")
    error_history = state.get("error_history", [])
    if previous_error and previous_error != "None":
        if not error_history or error_history[-1] != previous_error:
            error_history.append(previous_error)
    
    formatted_error_history = "\n".join([f"{i+1}. {err}" for i, err in enumerate(error_history)]) if error_history else "None"

    prompt_template = get_report_generation_prompt()
    chain = prompt_template | smart_llm | StrOutputParser()

    try:
        messages = state.get("messages", [HumanMessage(content=state["user_query"])])
        previous_columns = state.get("columns", [])
        
        # Format previous columns for prompt
        prev_cols_str = ", ".join(previous_columns) if previous_columns else "None"
        
        response = chain.invoke(
            {
                "messages": messages,
                "schema": json.dumps(table_schemas, indent=2),
                "user_query": state["user_query"],
                "error_history": formatted_error_history,
                "previous_columns": prev_cols_str,
                "company_id": state["company_id"],
            }
        )
        
        log_agent_step("QueryGenerator", "Raw LLM Response", {"response": response, "length": len(response)})
        
        generated_query, scratchpad = extract_sql(response)
        
        # Table Name Correction
        replacements_made = 0
        for user_table, correct_table in schema_table_mapping.items():
            if user_table.lower() != correct_table.lower():
                pattern = re.compile(
                    rf"(\b|\[){re.escape(user_table)}(\b|\])", re.IGNORECASE
                )
                before = generated_query
                generated_query = pattern.sub(
                    lambda m: f"{m.group(1)}{correct_table}{m.group(2)}",
                    generated_query,
                )
                if before != generated_query:
                    replacements_made += 1
                    log_agent_step(
                        "QueryGenerator",
                        "Table name replacement",
                        {
                            "from": user_table,
                            "to": correct_table,
                        },
                    )
        
        # Validate SQL syntax
        is_valid, validation_error = validate_sql_syntax(generated_query)
        if not is_valid:
            log_agent_step("QueryGenerator", "SQL Validation Warning (proceeding anyway)", {"error": validation_error})
            # return {
            #     **state,
            #     "generated_query": generated_query, # Persist for debugging
            #     "error": f"SQL validation failed: {validation_error}",
            #     "iteration_count": iteration + 1,
            # }

        return {
            **state,
            "generated_query": generated_query,
            "iteration_count": iteration,
            "error": None,
            "error_history": error_history,
            "last_scratchpad": scratchpad,
        }
    except Exception as e:
        return {**state, "error": str(e), "iteration_count": iteration + 1}

# Node 3: Report Validator (Executor)
def query_validator(state: AgentState) -> AgentState:
    # If there is already an error (e.g., from syntax check), pass it through
    if state.get("error"):
        return state

    query = state.get("generated_query", "").strip()
    iteration = state.get("iteration_count", 0) + 1
    
    if not query:
        return {**state, "error": "Empty query", "iteration_count": iteration}

    # Security Check
    query_upper = query.upper()
    dangerous_operations = [
        "DROP TABLE ", "DROP DATABASE ", "TRUNCATE TABLE ", "DELETE FROM ",
        "UPDATE ", "INSERT INTO ", "EXEC ", "EXECUTE ", "SHUTDOWN"
    ]
    if any(op in query_upper for op in dangerous_operations):
        error_msg = "Query contains potentially dangerous operations and was blocked"
        log_agent_step("QueryValidator", "Security Block", {"query": query}, level="WARNING")
        return {
            **state,
            "error": error_msg,
            "iteration_count": iteration
        }

    # Extract all variables used in the query
    variables = set(re.findall(r'@(\w+)', query))
    
    # Create dummy declarations for validation
    declarations = []
    for var in variables:
        if var.upper() in ['FROMDATE', 'TODATE']:
            declarations.append(f"DECLARE @{var} DATE = '2024-01-01';")
        else:
            # Default to NULL for optional filters
            declarations.append(f"DECLARE @{var} VARCHAR(MAX) = NULL;")
            
    # Inject TOP 5 for validation safety
    # We want to run the query but limit results to avoid performance hit
    # Instead of fragile regex replacement for TOP 5, use SET ROWCOUNT
    
    # Combine declarations with the modified query
    # Note: SET ROWCOUNT limits the result set for SELECT statements
    validation_query = "\n".join(declarations) + "\nSET ROWCOUNT 5;\n" + query + "\nSET ROWCOUNT 0;"
    
    log_agent_step("QueryValidator", "Validating query with dummy variables and SET ROWCOUNT", {"validation_query": validation_query})

    try:
        with engine.connect() as connection:
            # Execute the validation query
            connection.execute(text(validation_query))
            
        log_agent_step("QueryValidator", "Validation successful")
        return {
            **state,
            "error": None,
            "iteration_count": iteration,
            "generated_query": query # We return the ORIGINAL query (template), not the validation query
        }
    except Exception as e:
        log_error("QueryValidator", e)
        return {
            **state,
            "error": str(e),
            "iteration_count": iteration
        }

def query_executor(state: AgentState) -> AgentState:
    """Executes the generated SQL query."""
    if state.get("error"):
        return state

    query = state.get("generated_query")
    if not query:
        return {**state, "error": "No query generated"}
    
    try:
        from langgraph_sql_agent_chat import engine
        from sqlalchemy import text
        import re
        
        # Extract parameters used in SQL
        params_in_sql = set(re.findall(r'@([a-zA-Z0-9_]+)', query))
        
        # Prepare execution params: default to None (NULL)
        execution_params = {param: None for param in params_in_sql}
        
        # Ensure CompanyId is bound if used
        if 'CompanyId' in params_in_sql and state.get("company_id"):
             execution_params['CompanyId'] = state["company_id"]
             
        with engine.connect() as connection:
            stmt = text(query)
            result = connection.execute(stmt, execution_params)
            
            # columns = list(result.keys())
            raw_columns = list(result.keys())
            columns = [c.replace(" ", "") for c in raw_columns]  # e.g. Invoice No -> InvoiceNo
            
            rows = [dict(zip(columns, row)) for row in result.fetchall()]
            return {**state, "execution_results": rows, "columns": columns}

    except Exception as e:
        return {**state, "error": f"Execution failed: {str(e)}"}

def should_correct(state: AgentState) -> str:
    if state.get("error"):
        if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
            return END
        return "query_generator"
    return "query_executor"

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("table_selector", table_selector)
workflow.add_node("query_generator", query_generator)
workflow.add_node("query_validator", query_validator)
workflow.add_node("query_executor", query_executor)

workflow.set_entry_point("table_selector")
workflow.add_edge("table_selector", "query_generator")
workflow.add_edge("query_generator", "query_validator")

# Use conditional edges properly
workflow.add_conditional_edges(
    "query_validator",
    should_correct,
    {
        "query_generator": "query_generator",
        "query_executor": "query_executor",
        END: END
    }
)
workflow.add_edge("query_executor", END)

app = workflow.compile(checkpointer=MemorySaver())

def run_report_generation(query: str, company_id: int = 1, thread_id: str = None) -> Dict[str, Any]:
    
    # Generate a thread ID if one wasn't provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "user_query": query,
        "company_id": company_id,
        "selected_tables": [],
        "table_schemas": {},
        "generated_query": "",
        "error": None,
        "iteration_count": 0,
        "max_iterations": 3,
        "description_a": DESCRIPTION_A,
        "messages": [HumanMessage(content=query)],
    }
    
    try:
        # Use simple invoke. Logic handles appending messages via reducer.
        result = app.invoke(inputs, config=config)
        return {
            "sql_query": result.get("generated_query"),
            "error": result.get("error"),
            "scratchpad": result.get("last_scratchpad") or (result['messages'][-1].content if result.get('messages') else ""),
            "thread_id": thread_id,
            "data": result.get("execution_results"),
            "columns": result.get("columns")
        }
    except Exception as e:
        return {"error": str(e), "thread_id": thread_id}
