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
    declare_vars = re.findall(r'DEclare\s+@(\w+)', sql, re.IGNORECASE)
    
    # Check for variable usage without declaration
    used_vars = re.findall(r'@(\w+)', sql)
    declared_vars = set(v.upper() for v in declare_vars)
    
    # Common system variables to ignore
    system_vars = {'IDENTITY', 'ROWCOUNT', 'ERROR', 'TRANCOUNT', 'VERSION', 'SPID', 'FETCH_STATUS'}
    
    for var in used_vars:
        if var.upper() in system_vars or var.upper().startswith('@@'):
            continue
            
        if var.upper() not in declared_vars:
            errors.append(f"Variable @{var} is used but not declared")
    
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
    scratchpad_match = re.search(
        r"<scratchpad>(.*?)</scratchpad>", llm_output, re.DOTALL | re.IGNORECASE
    )
    scratchpad_text = scratchpad_match.group(1).strip() if scratchpad_match else None
    
    sql_match = re.search(r"<sql>(.*?)</sql>", llm_output, re.DOTALL | re.IGNORECASE)
    if sql_match:
        sql = sql_match.group(1).strip()
    else:
        # Fallback extraction
        code_blocks = re.findall(r"```(?:sql)?\s*(.*?)\s*```", llm_output, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            sql = "\n".join(block.strip() for block in code_blocks)
        else:
            sql = llm_output.strip()

    # Cleanup
    sql = re.sub(r"^\s*--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    if not sql.endswith(";"):
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

1. **UNDERSTAND THE GOAL:**
   - The user wants a report (e.g., "Sales by Customer", "Job Revenue").
   - The query MUST be dynamic, allowing filtering by Date Range AND any selected column.

2. **ANALYZE THE SCHEMA:**
   - Review the available tables and their columns.
   - Identify which tables contain the required data.
   - Note any relationships between tables (foreign keys).

3. **COLUMN SELECTION PRINCIPLES (CRITICAL):**
   - **Distinguish Keys:**
     - **Surrogate Keys:** (e.g., GUIDs, auto-increments, IDs) are for system use. DO NOT include them unless the user explicitly asks for an ID.
     - **Natural Keys:** (e.g., Names, Codes, Titles) are for humans. ALWAYS prefer these over surrogate keys.
   - **Audit Data:** Columns like `CreatedBy`, `ModifiedDate`, `ConcurrencyStamp` are metadata. Exclude them.
   - **User Intent:**
     - If the user asks for "Jobs", they want the *Job Name*, *Date*, and *Status*. They do NOT want the `JobId` or `CompanyId`.
     - If the user asks for "Clients", they want the *Customer Name*, not the `CustomerId`.
   - **Readability:**
     - Prefer descriptive text columns over numeric codes.
     - Limit output to the most relevant 6-8 columns.

4. **MANDATORY FILTERING LOGIC (N+2 FILTERS):**
   - **Date Range:** You MUST include `@FromDate` and `@ToDate` filters on the primary date column (e.g., JobDate, InvoiceDate).
   - **Column Filters:** For EVERY column in your `SELECT` clause, you MUST add an optional filter in the `WHERE` clause.
   - **Pattern:** `AND (@ParamName IS NULL OR Table.Column = @ParamName)`

5. **QUERY STRUCTURE:**
   ```sql
   SELECT
       t.Column1 AS [Friendly Name 1],
       t.Column2 AS [Friendly Name 2],
       ...
   FROM Table t
   ...
   WHERE
       t.CompanyId = 1 -- Multi-tenancy
       AND (t.IsDeleted = 0 OR t.IsDeleted IS NULL) -- Soft delete
       -- MANDATORY DATE FILTER
       AND t.DateColumn BETWEEN @FromDate AND @ToDate
       -- DYNAMIC COLUMN FILTERS (One for EACH selected column)
       AND (@FriendlyName1 IS NULL OR t.Column1 = @FriendlyName1)
       AND (@FriendlyName2 IS NULL OR t.Column2 = @FriendlyName2)
       ...
   ```

6. **CRITICAL RULES:**
   - **DO NOT DECLARE VARIABLES:** The frontend/API will handle declarations. Your output should START with `SELECT`.
   - **Parameter Names:** Use `@` + the alias name (or column name if no alias) for parameters.
   - **Multi-Tenancy:** ALWAYS filter by `CompanyId = {company_id}`.
   - **Soft Deletes:** ALWAYS filter `IsDeleted = 0`.
   - **Recurring Jobs (CRITICAL):** When querying recurring jobs (e.g., Visits, Schedules), the 'Start Time' and 'End Time' in the database might be the *series* start/end. You MUST construct the actual instance datetime using the `VisitDate` (or equivalent) combined with the time component of the start/end columns.
     - Example: `DATEADD(day, DATEDIFF(day, '19000101', VisitDate), CAST(StartTime AS DATETIME))`
     - Use this computed date for the SELECT clause AND the `@FromDate/@ToDate` filter.
   - **No Technical Columns:** ABSOLUTELY NO technical/ID columns in SELECT unless explicitly requested.
   - **Joins:** Include proper JOIN conditions between tables.
   - **ENUM/LOOKUP COLUMNS (CRITICAL):** Check the schema description for any column that defines a mapping (e.g., "1=Created, 2=Scheduled").
     - **FILTERING:** Use the **INTEGER** value (e.g., `JobStatus IN (2, 3)`).
     - **SELECTING:** Select the raw column (e.g., `JobStatus`). The system will automatically map it to the human-readable name.

7. **COMMON PITFALLS TO AVOID:**
   - Don't compare string IDs with integer IDs.
   - Don't use string concatenation for multiple IDs.
   - Ensure data types match in WHERE conditions and JOINs.

8. **THINK STEP BY STEP:**
   1. What information is the user asking for?
   2. Which tables contain this information?
   3. How should these tables be joined?
   4. What is the primary date column for the `@FromDate`/`@ToDate` filter?
   5. For each selected column, have I added the corresponding dynamic filter?
   6. Did I handle recurring job dates correctly?
   7. Did I exclude technical columns?

9. **OUTPUT FORMAT:**
   - Return ONLY the SQL query inside <sql> tags.
   - Include a <scratchpad> for your thinking.

DATABASE SCHEMA:
{schema}

PREVIOUS ERRORS:
{error_history}
""",
            ),
            ("human", "{user_query}"),
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
        response = chain.invoke(
            {
                "schema": json.dumps(table_schemas, indent=2),
                "user_query": state["user_query"],
                "error_history": formatted_error_history,
                "company_id": state["company_id"],
            }
        )
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
            return {
                **state,
                "error": f"SQL syntax validation failed: {validation_error}",
                "iteration_count": iteration,
            }

        return {
            **state,
            "generated_query": generated_query,
            "iteration_count": iteration,
            "error": None,
            "error_history": error_history,
            "last_scratchpad": scratchpad,
        }
    except Exception as e:
        return {**state, "error": str(e)}

# Node 3: Report Validator (Executor)
def query_validator(state: AgentState) -> AgentState:
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
    validation_sql = query
    if "SELECT" in validation_sql.upper():
        # Handle SELECT DISTINCT
        if "SELECT DISTINCT" in validation_sql.upper():
            validation_sql = re.sub(r"SELECT\s+DISTINCT", "SELECT DISTINCT TOP 5", validation_sql, count=1, flags=re.IGNORECASE)
        else:
            validation_sql = re.sub(r"SELECT", "SELECT TOP 5", validation_sql, count=1, flags=re.IGNORECASE)

    # Combine declarations with the modified query
    validation_query = "\n".join(declarations) + "\n" + validation_sql
    
    log_agent_step("QueryValidator", "Validating query with dummy variables and TOP 5", {"validation_query": validation_query})

    try:
        with engine.connect() as connection:
            # Execute the validation query
            connection.execute(text(validation_query))
            
        log_agent_step("QueryValidator", "Validation successful")
        return {
            **state,
            "error": None,
            "iteration_count": iteration,
            # We return the ORIGINAL query (template), not the validation query
            "generated_query": query 
        }
    except Exception as e:
        log_error("QueryValidator", e)
        return {
            **state,
            "error": str(e),
            "iteration_count": iteration
        }

def should_correct(state: AgentState) -> str:
    if state.get("error"):
        if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
            return END
        return "query_generator"
    return END

# Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("table_selector", table_selector)
workflow.add_node("query_generator", query_generator)
workflow.add_node("query_validator", query_validator)

workflow.set_entry_point("table_selector")
workflow.add_edge("table_selector", "query_generator")
workflow.add_edge("query_generator", "query_validator")
workflow.add_conditional_edges("query_validator", should_correct, {"query_generator": "query_generator", END: END})

app = workflow.compile()

def run_report_generation(query: str, company_id: int = 1) -> Dict[str, Any]:
    inputs = {
        "user_query": query,
        "company_id": company_id,
        "messages": [HumanMessage(content=query)],
        "max_iterations": 3,
    }
    
    try:
        result = app.invoke(inputs)
        return {
            "sql_query": result.get("generated_query"),
            "error": result.get("error"),
            "scratchpad": result.get("last_scratchpad")
        }
    except Exception as e:
        return {"error": str(e)}
