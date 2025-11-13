import json
import os
import re
import logging
import uuid
import time
from datetime import datetime
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Logging configuration (integrated from provided module)
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
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def configure_logging():
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"agent_{timestamp}.log")
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-10s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
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
        return '[' + ', '.join(_format_data(item) for item in data) + ']'
    elif isinstance(data, dict):
        return '{' + ', '.join(f'"{k}": {_format_data(v)}' for k, v in data.items()) + '}'
    else:
        return str(data)

def log_agent_step(
    agent_name: str,
    message: str,
    data: Optional[Dict[str, Any]] = None,
    level: str = "INFO"
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    log_msg = f"{agent_name}: {message}"
    
    if data is not None and len(data) > 0:
        try:
            formatted_data = '\n'.join(f"{k}: {_format_data(v)}" for k, v in data.items())
            log_msg = f"{log_msg}\n{formatted_data}"
        except Exception as e:
            logger.warning(f"Failed to format log data: {e}")
    
    logger.log(log_level, log_msg)

def log_error(
    agent_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR"
) -> None:
    error_data = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        **({} if context is None else context)
    }
    log_agent_step(agent_name, "Error occurred", error_data, level)

load_dotenv()  # Load .env for OPENAI_API_KEY

# Schema Cache
class SchemaCache:
    def __init__(self):
        self._cache = {}
        self._last_modified = {}
    
    def get(self, file_path: str) -> Any:
        try:
            current_mtime = os.path.getmtime(file_path)
            
            if (file_path not in self._cache or 
                file_path not in self._last_modified or 
                self._last_modified[file_path] < current_mtime):
                
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.endswith('.json'):
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
class ModelConfig:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "gpt-5-mini")
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", "0"))
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    
    def get_llm(self):
        return ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.openai_api_key
        )

try:
    model_config = ModelConfig()
    llm = model_config.get_llm()

    try:
        response = llm.invoke("Test message")
        print(f"Successfully connected to {model_config.model_name}!")
    except Exception as e:
        print(f"Error connecting to OpenAI API: {str(e)}")
        print("Please check your API key and internet connection.")
        raise

except Exception as e:
    print(f"Error initializing language model: {str(e)}")
    print("Please make sure you have set up your OpenAI API key correctly.")
    print("You can get an API key from: https://platform.openai.com/api-keys")
    print("Then set it in your environment variables or in a .env file:")
    print("OPENAI_API_KEY=your-api-key-here")
    raise

# DB Connection
try:
    driver = "ODBC Driver 18 for SQL Server"
    DB_SERVER = os.getenv("DB_SERVER")
    DB_DATABASE = os.getenv("DB_DATABASE")
    DB_UID = os.getenv("DB_UID")
    DB_PWD = os.getenv("DB_PWD")

    print(f"Attempting to connect to server: {DB_SERVER}")
    print(f"Database: {DB_DATABASE}")
    try:
        odbc_params = (
            f"DRIVER={{{driver}}};"
            f"SERVER={DB_SERVER};"
            f"DATABASE={DB_DATABASE};"
            f"UID={DB_UID};"
            f"PWD={DB_PWD};"
            # "Encrypt=yes;"
            "TrustServerCertificate=yes;"
            "Trusted_Connection=yes;"
            # "Connection Timeout=15;"
            # "Login Timeout=15;"
        )
        DATABASE_URI = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"
        engine = create_engine(DATABASE_URI, pool_pre_ping=True)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Successfully connected to the database!")

    except Exception as e:
        print(f"\nFirst connection attempt failed. Error: {str(e)}")
        print("\nTrying alternative connection method...")

        try:
            server = DB_SERVER.replace("tcp:", "").split(",")[0]
            port = DB_SERVER.split(",")[1] if "," in DB_SERVER else "1433"

            odbc_params = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server},{port};"
                f"SERVER={server};"
                f"DATABASE={DB_DATABASE};"
                f"UID={DB_UID};"
                f"PWD={DB_PWD};"
                #"Encrypt=yes;"
                "TrustServerCertificate=yes;"
                "Trusted_Connection=yes;"
                #"Connection Timeout=15;"
            )
            DATABASE_URI = f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_params)}"
            engine = create_engine(DATABASE_URI, pool_pre_ping=True)

            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Successfully connected to the database using alternative connection method!")

        except Exception as alt_e:
            print(f"\nAlternative connection attempt also failed. Error: {str(alt_e)}")
            print("\nPlease check the following:")
            print("1. Is the SQL Server running and accessible?")
            print("2. Are the server name, database name, username, and password correct?")
            print("3. Is the SQL Server configured to accept remote connections?")
            print("4. Is the firewall allowing connections to the SQL Server port (default 1433)?")
            print("5. Are you using the correct ODBC driver?")
            print("\nYou can set environment variables to override the default connection settings:")
            print("set DB_SERVER=your_server_name,port")
            print("set DB_DATABASE=your_database_name")
            print("set DB_UID=your_username")
            print("set DB_PWD=your_password")
            raise

except ImportError as ie:
    print("\nError: Required Python packages not found. Please install them with:")
    print("pip install pyodbc sqlalchemy python-dotenv")
    raise

except Exception as e:
    print(f"\nFailed to establish database connection. Error: {str(e)}")
    print("\nPlease ensure that:")
    print("1. SQL Server is running and accessible")
    print("2. The specified database and credentials are correct")
    print("3. The SQL Server allows remote connections")
    print("4. Your firewall allows connections to the SQL Server port")
    print("5. You have the correct ODBC driver installed")
    raise

def extract_sql(llm_output: str) -> str:
    code_block_pattern = r"```(?:sql)?\s*(.*?)\s*```"
    code_block_match = re.search(code_block_pattern, llm_output, re.DOTALL | re.IGNORECASE)

    if code_block_match:
        sql = code_block_match.group(1).strip()
    else:
        sql_pattern = r"(?i)(?:^|\n)(WITH\s+[^;]+|SELECT\s+[^;]+)(?:;|$|\n\n)"
        sql_match = re.search(sql_pattern, llm_output, re.DOTALL)
        if not sql_match:
            sql_pattern = r"(?i)(?:^|\n)((?:WITH|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE)\s+[\s\S]+?)(?:;|$|\n\n)"
            sql_match = re.search(sql_pattern, llm_output)

        sql = sql_match.group(1).strip() if sql_match else llm_output.strip()

    sql = re.sub(r"^\s*--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"\n\s*\n", "\n", sql)
    sql = sql.strip()

    if not sql.endswith(";"):
        sql += ";"

    log_agent_step("SQLExtractor", "Extracted SQL query", {"sql_query": sql})
    return sql

def sanitize_sql_input(input_str: str) -> str:
    if not input_str:
        return ""
    sanitized = re.sub(r'[;\-\-\n]', ' ', input_str)
    sanitized = re.sub(r'/\*.*?\*/', '', sanitized)
    return sanitized.strip()

def get_sql_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert. Generate a T-SQL query to answer the user's question.
        
        CRITICAL RULES:
        1. Use ONLY these exact table names: {table_list}
        2. Use the EXACT column names as shown in the schema (case-sensitive)
        3. For date/time filters, use proper date functions with explicit CONVERT(DATE, ...)
        4. Include proper JOIN conditions between tables
        5. Only return the SQL query, nothing else
        6. NEVER use string concatenation for ID comparisons - always use proper JOINs or IN with individual values
        7. When using IN with subqueries, ensure the data types match exactly
        8. For ID fields, always use the correct data type (usually INT or UNIQUEIDENTIFIER)
        9. When joining tables, use the exact column names as they appear in the schema
        
        {date_handling_instruction}
        
        COMMON PITFALLS TO AVOID:
        - Don't compare string IDs with integer IDs
        - Don't use string concatenation for multiple IDs
        - Always use proper JOINs instead of IN with string concatenation
        - Ensure data types match in WHERE conditions and JOINs
        
        DATABASE SCHEMA:
        {table_schemas}
        
        USER QUESTION: {user_query}
        
        PREVIOUS ERROR (if any): {previous_error}
        
        YOUR T-SQL QUERY (ONLY the SQL, no markdown or code blocks):"""),
        ("human", "{user_query}")
    ])

# State definition
class AgentState(TypedDict):
    run_id: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    selected_tables: List[str]
    table_schemas: Dict[str, Any]
    generated_query: str
    execution_result: Any
    error: str
    is_empty_result: bool
    iteration_count: int
    max_iterations: int
    description_a: str

# Node 1: Table Selector Agent
def table_selector(state: AgentState) -> AgentState:
    try:
        log_agent_step("TableSelector", "Starting table selection", {
            "user_query": state["user_query"],
            "iteration": state.get("iteration_count", 0)
        })
        
        description_a = state.get("description_a", DESCRIPTION_A)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a table selector agent. Use the provided database overview to identify the most relevant tables for the user's query. 
            Focus on key entities like jobs, schedule, users, clients, etc.
            
            DATABASE OVERVIEW:
            {overview}
            
            Output ONLY a JSON object with key "tables" containing a list of table names.
            Example: {{"tables": ["Job", "Users", "Clients"]}}"""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        chain = prompt | llm | JsonOutputParser()
        response = chain.invoke({
            "messages": state["messages"],
            "overview": description_a
        })
        
        selected_tables = response.get("tables", [])
        log_agent_step("TableSelector", "Selected tables", {"selected_tables": selected_tables})
        
        return {
            **state,
            "selected_tables": selected_tables,
            "messages": state["messages"] + [AIMessage(content=f"Selected tables: {', '.join(selected_tables)}")]
        }
        
    except Exception as e:
        log_error("TableSelector", e, "Error in table selection")
        return {
            **state,
            "error": str(e),
            "messages": state["messages"] + [AIMessage(content=f"Error selecting tables: {str(e)}")]
        }

def load_table_schema(table: str, description_b: Dict) -> Optional[Dict]:
    exact = next((t for t in description_b if t.lower() == table.lower()), None)
    if exact and "columns" in (info := description_b[exact]):
        columns_data = info["columns"]
        if isinstance(columns_data, list):
            columns = [col.get("Column Name", "") for col in columns_data if isinstance(col, dict) and "Column Name" in col]
            return {"description": info.get("description", ""), "columns": {col: {} for col in columns if col}}
    return None

# Node 2: Query Generator Agent
def query_generator(state: AgentState) -> AgentState:
    run_id = state.get("run_id", str(uuid.uuid4()))
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    is_retry = iteration > 0
    
    if iteration >= max_iterations:
        error_msg = f"Maximum number of query generation attempts ({max_iterations}) reached"
        log_agent_step("QueryGenerator", error_msg, {"run_id": run_id, "iteration": iteration}, level="ERROR")
        return {
            **state,
            "error": error_msg,
            "messages": state["messages"] + [AIMessage(content=error_msg)]
        }
    
    log_agent_step("QueryGenerator", f"{'Retry ' if is_retry else ''}Generating query (Iteration: {iteration + 1}/{max_iterations})", {
        "run_id": run_id,
        "user_query": state["user_query"],
        "selected_tables": state.get("selected_tables", []),
        "previous_error": state.get("error", "None"),
        "is_empty_result": state.get("is_empty_result", False)
    })
    
    try:
        selected_tables = state.get("selected_tables", [])
        log_agent_step("QueryGenerator", "Starting SQL query generation", {
            "run_id": run_id,
            "tables": selected_tables,
            "previous_error": state.get("error", "None"),
            "iteration": iteration
        })
        
        table_schemas = {}
        schema_table_mapping = {}
        
        try:
            first_table = next(iter(DESCRIPTION_B.values())) if DESCRIPTION_B else {}
            log_agent_step("QueryGenerator", "Schema structure sample", {
                "first_table_keys": list(first_table.keys()) if isinstance(first_table, dict) else "Not a dict",
                "has_columns": "columns" in first_table if isinstance(first_table, dict) else False,
                "columns_type": type(first_table.get("columns")).__name__ if isinstance(first_table, dict) else "N/A"
            }, level="DEBUG")
            
            available_tables = list(DESCRIPTION_B.keys())
            log_agent_step("QueryGenerator", "Processing tables", {
                "available_tables_count": len(available_tables),
                "selected_tables": selected_tables
            })
            
            for table in selected_tables:
                if not table:
                    continue
                    
                schema = load_table_schema(table, DESCRIPTION_B)
                if schema:
                    exact_match = next((t for t in available_tables if t.lower() == table.lower()), None)
                    table_schemas[exact_match] = schema
                    schema_table_mapping[table] = exact_match
                    log_agent_step("QueryGenerator", "Table schema loaded", {
                        "input_table": table,
                        "mapped_to": exact_match,
                        "column_count": len(schema["columns"])
                    })
                else:
                    log_agent_step("QueryGenerator", "Table not found in schema", {"table": table}, level="WARNING")
                    
        except Exception as e:
            error_msg = f"Error processing schema: {str(e)}"
            log_agent_step("QueryGenerator", error_msg, {"error_type": type(e).__name__}, level="ERROR")
            return {
                **state,
                "error": error_msg,
                "messages": state["messages"] + [AIMessage(content=error_msg)]
            }
        
        if not table_schemas:
            error_msg = f"No valid table schemas found for selected tables: {selected_tables}"
            log_agent_step("QueryGenerator", error_msg, level="ERROR")
            return {
                **state,
                "error": error_msg,
                "messages": state["messages"] + [AIMessage(content=error_msg)]
            }
        
        table_list = ", ".join(f'"{t}"' for t in table_schemas.keys())
        
        date_handling_instruction = """
        IMPORTANT DATE HANDLING RULES:
        1. Always use CONVERT(DATE, column_name) when comparing with GETDATE() or other date functions
        2. For date ranges, use: 
           WHERE CONVERT(DATE, date_column) BETWEEN CONVERT(DATE, GETDATE()) AND DATEADD(day, 7, CONVERT(DATE, GETDATE()))
        3. Handle NULL dates properly with IS NULL/IS NOT NULL
        4. When using date functions, ensure the format is compatible with SQL Server
        """
        
        prompt_template = get_sql_generation_prompt()
        chain = prompt_template | llm | StrOutputParser()
        
        response = chain.invoke({
            "table_list": table_list,
            "table_schemas": json.dumps(table_schemas, indent=2),
            "user_query": state["user_query"],
            "previous_error": state.get("error", "None"),
            "date_handling_instruction": date_handling_instruction
        })
        
        log_agent_step("QueryGenerator", "Received response from LLM", {
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "response_length": len(response)
        })
        
        generated_query = extract_sql(response)
        
        original_query = generated_query
        if "```" in generated_query:
            parts = generated_query.split("```")
            for part in parts[1:]:
                if any(keyword in part.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE"]):
                    generated_query = part
                    break
        
        generated_query = re.sub(r'^sql\s*\n', '', generated_query, flags=re.IGNORECASE)
        generated_query = generated_query.strip('` \n')
        
        log_agent_step("QueryGenerator", "Query cleaning results", {
            "original": original_query[:200] + ("..." if len(original_query) > 200 else ""),
            "cleaned": generated_query[:200] + ("..." if len(generated_query) > 200 else "")
        })
        
        if not generated_query.strip():
            error_msg = "Generated SQL query is empty"
            log_agent_step("QueryGenerator", error_msg, level="ERROR")
            return {
                **state,
                "error": error_msg,
                "messages": state["messages"] + [AIMessage(content=error_msg)]
            }
            
        replacements_made = 0
        for user_table, correct_table in schema_table_mapping.items():
            if user_table.lower() != correct_table.lower():
                pattern = re.compile(fr'(\b|\[){re.escape(user_table)}(\b|\])', re.IGNORECASE)
                before = generated_query
                generated_query = pattern.sub(lambda m: f"{m.group(1)}{correct_table}{m.group(2)}", generated_query)
                if before != generated_query:
                    replacements_made += 1
                    log_agent_step("QueryGenerator", "Table name replacement", {
                        "from": user_table,
                        "to": correct_table,
                        "query_after": generated_query[:200] + ("..." if len(generated_query) > 200 else "")
                    })
        
        log_agent_step("QueryGenerator", f"Table name replacements made: {replacements_made}")
        
        new_iteration_count = state["iteration_count"]  # No +1 here
        
        log_agent_step("QueryGenerator", f"Generated SQL query (Iteration: {new_iteration_count + 1})", {
            "run_id": run_id,
            "sql_query": generated_query,
            "tables_used": list(table_schemas.keys())
        })
        
        return {
            **state,
            "table_schemas": table_schemas,
            "generated_query": generated_query,
            "iteration_count": new_iteration_count,
            "error": None,
            "messages": state["messages"] + [AIMessage(content=f"Generated SQL (iteration {new_iteration_count + 1}): {generated_query}")]
        }
    except Exception as e:
        log_agent_step("QueryGenerator", "Error generating SQL query", {"error": str(e)}, level="ERROR")
        return {
            **state,
            "error": str(e),
            "messages": state["messages"] + [AIMessage(content=f"Error generating SQL query: {str(e)}")]
        }

def execute_with_retry(execute_fn, max_retries=3, initial_delay=1.0, backoff_factor=2.0):
    retry_info = {'attempts': 0, 'delays': [], 'errors': [], 'start_time': time.time(), 'end_time': None, 'total_duration': None}
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        retry_info['attempts'] += 1
        start_time = time.time()
        
        try:
            result, error = execute_fn()
            
            if not error:
                retry_info['end_time'] = time.time()
                retry_info['total_duration'] = retry_info['end_time'] - retry_info['start_time']
                return result, None, retry_info
                
            retry_info['errors'].append({'attempt': attempt + 1, 'error': str(error), 'error_type': type(error).__name__, 'timestamp': time.time(), 'execution_time': time.time() - start_time})
            
            if attempt >= max_retries:
                return None, error, retry_info
                
            time.sleep(delay)
            retry_info['delays'].append(delay)
            delay *= backoff_factor
            
        except Exception as e:
            error = e
            retry_info['errors'].append({'attempt': attempt + 1, 'error': str(e), 'error_type': type(e).__name__, 'timestamp': time.time(), 'execution_time': time.time() - start_time, 'is_exception': True})
            
            if attempt >= max_retries:
                return None, error, retry_info
                
            time.sleep(delay)
            retry_info['delays'].append(delay)
            delay *= backoff_factor
    
    return None, error, retry_info

def query_executor(state: AgentState) -> AgentState:
    run_id = state.get("run_id", str(uuid.uuid4()))
    iteration = state.get("iteration_count", 0) + 1
    max_iterations = state.get("max_iterations", 3)
    start_time = time.time()
    execution_metrics = {
        'start_time': start_time,
        'end_time': None,
        'duration_seconds': None,
        'retry_attempts': 0,
        'query_preview': None,
        'row_count': 0,
        'columns': [],
        'execution_plan': None,
        'warnings': [],
        'retry_info': None
    }
    
    query = state.get("generated_query", "").strip()
    execution_metrics['query_preview'] = query[:500]
    
    if not query:
        error_msg = "No SQL query provided for execution"
        log_agent_step("QueryExecutor", "Error: No query to execute", {
            "run_id": run_id,
            "error": error_msg,
            "iteration": iteration,
            "execution_metrics": execution_metrics
        }, level="ERROR")
        return {
            **state,
            "execution_result": None,
            "error": error_msg,
            "is_empty_result": True,
            "iteration_count": iteration,
            "execution_metrics": execution_metrics,
            "messages": state["messages"] + [AIMessage(content=f"Error: {error_msg}")]
        }
        
    dangerous_operations = ["DROP ", "DELETE ", "TRUNCATE ", "UPDATE ", "INSERT ", "EXEC ", "EXECUTE ", "DECLARE ", "SHUTDOWN"]
    if any(op in query.upper() for op in dangerous_operations):
        error_msg = "Query contains potentially dangerous operations and was blocked"
        execution_metrics['warnings'].append({
            'type': 'security_block',
            'message': 'Query contained blocked SQL operations',
            'operations': [op.strip() for op in dangerous_operations if op in query.upper()]
        })
        log_agent_step("QueryExecutor", "Security: Blocked potentially dangerous query", {
            "run_id": run_id,
            "error": error_msg,
            "iteration": iteration,
            "execution_metrics": execution_metrics
        }, level="WARNING")
        return {
            **state,
            "execution_result": None,
            "error": error_msg,
            "is_empty_result": True,
            "iteration_count": iteration,
            "execution_metrics": execution_metrics,
            "messages": state["messages"] + [AIMessage(content=f"Security Error: {error_msg}")]
        }
    
    log_agent_step("QueryExecutor", "Starting SQL query execution", {
        "run_id": run_id,
        "query_preview": execution_metrics['query_preview'],
        "tables_involved": state.get("selected_tables", []),
        "iteration": iteration,
        "max_iterations": max_iterations,
        "execution_metrics": execution_metrics
    })
    
    def execute_query():
        with engine.connect() as connection:
            with connection.begin():
                result = connection.execute(text(query))
                
                columns = []
                if hasattr(result, 'keys'):
                    columns = list(result.keys())
                elif hasattr(result, '_metadata') and hasattr(result._metadata, 'keys'):
                    columns = list(result._metadata.keys)
                
                rows = result.fetchall()
                execution_result = [dict(zip(columns, row)) for row in rows]
                
                execution_metrics.update({
                    'row_count': len(execution_result),
                    'columns': columns,
                    'is_empty_result': len(execution_result) == 0,
                    'execution_plan': str(columns)
                })
                
                return execution_result, None
                
    execution_result, error, retry_info = execute_with_retry(execute_query)
    execution_metrics.update({
        'end_time': time.time(),
        'duration_seconds': time.time() - start_time,
        'retry_info': retry_info,
        'retry_attempts': len(retry_info['errors']) if retry_info else 0
    })
            
    if error:
        error_msg = f"Error executing query: {str(error)}"
        execution_metrics['error'] = {'message': str(error), 'type': type(error).__name__, 'retry_attempts': execution_metrics['retry_attempts']}
        
        if "There is already an open DataReader" in str(error):
            error_msg = "Database connection error: Multiple active result sets detected. Please try again."
            execution_metrics['warnings'].append({
                'type': 'connection_warning',
                'message': 'Multiple active result sets detected',
                'suggestion': 'Consider using MARS in your connection string'
            })
        elif "timeout" in str(error).lower() or "timed out" in str(error).lower():
            error_msg = "Query execution timed out. Please try a more specific query."
            execution_metrics['warnings'].append({'type': 'timeout', 'message': 'Query timeout'})
        
        log_agent_step("QueryExecutor", f"Query execution failed: {error_msg}", {
            "run_id": run_id,
            "error": str(error),
            "error_type": type(error).__name__,
            "iteration": iteration,
            "execution_metrics": execution_metrics
        }, level="ERROR")
        
        status_msg = error_msg
    else:
        row_count = execution_metrics['row_count']
        duration = execution_metrics['duration_seconds']
        
        if execution_metrics['is_empty_result']:
            status_msg = "Query executed successfully but returned no results."
        else:
            status_msg = f"Query executed successfully in {duration:.2f}s. Returned {row_count} row{'s' if row_count != 1 else ''}."
        
        log_agent_step("QueryExecutor", "Query executed successfully", {
            "run_id": run_id,
            "row_count": row_count,
            "duration_seconds": duration,
            "iteration": iteration,
            "execution_metrics": execution_metrics
        })
    
    return {
        **state,
        "execution_result": execution_result,
        "error": str(error) if error else None,
        "is_empty_result": execution_metrics.get('is_empty_result', True),
        "iteration_count": iteration,
        "execution_metrics": execution_metrics,
        "messages": state["messages"] + [AIMessage(content=status_msg)]
    }

def should_correct(state: AgentState) -> str:
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    is_empty = state.get("is_empty_result", True)
    error = state.get("error")
    has_error = bool(error)
    
    log_agent_step("CorrectionHandler", "Deciding if correction is needed", {
        "iteration": iteration,
        "max_iterations": max_iterations,
        "has_error": has_error,
        "is_empty_result": is_empty,
        "previous_query": state.get("generated_query", "")[:200] + ("..." if len(state.get("generated_query", "")) > 200 else ""),
        "error_message": error,
        "tables_involved": state.get("selected_tables", [])
    })
    
    if iteration >= max_iterations:
        log_agent_step("CorrectionHandler", "Max iterations reached, stopping correction loop", {"iteration": iteration})
        return END
    
    if has_error or is_empty:
        log_agent_step("CorrectionHandler", "Correction needed", {
            "reason": "Error in execution" if has_error else "Empty result set",
            "error": error,
            "current_iteration": iteration + 1,
            "decision": "retry"
        })
        return "query_generator"
    
    log_agent_step("CorrectionHandler", "No correction needed", {"decision": "end"})
    return END

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("table_selector", table_selector)
workflow.add_node("query_generator", query_generator)
workflow.add_node("query_executor", query_executor)

workflow.set_entry_point("table_selector")
workflow.add_edge("table_selector", "query_generator")
workflow.add_edge("query_generator", "query_executor")
workflow.add_conditional_edges(
    "query_executor",
    should_correct,
    {
        "query_generator": "query_generator",
        END: END
    }
)

app = workflow.compile()

def run_conversational_query(query: str, max_iterations: int = 3, recursion_limit: int = 10) -> Dict[str, Any]:
    run_id = str(uuid.uuid4())
    sanitized_query = sanitize_sql_input(query)
    
    log_agent_step("System", "Starting new conversational query", {
        "run_id": run_id,
        "query": sanitized_query,
        "max_iterations": max_iterations,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    initial_state = {
        "run_id": run_id,
        "messages": [HumanMessage(content=sanitized_query)],
        "user_query": sanitized_query,
        "description_a": DESCRIPTION_A,
        "selected_tables": [],
        "table_schemas": {},
        "generated_query": "",
        "execution_result": None,
        "error": None,
        "is_empty_result": False,
        "iteration_count": 0,
        "max_iterations": max_iterations
    }
    
    config = {"recursion_limit": recursion_limit}
    
    try:
        final_state = app.invoke(initial_state, config=config)
    except Exception as e:
        log_error("WorkflowExecution", e, {"query": query, "max_iterations": max_iterations})
        raise

    formatted_results = []
    if final_state["execution_result"] and not final_state.get("is_empty_result", False):
        try:
            if final_state["execution_result"] and isinstance(final_state["execution_result"][0], dict):
                formatted_results = [
                    {key: str(value) if value is not None else None 
                     for key, value in row.items()}
                    for row in final_state["execution_result"]
                ]
            else:
                formatted_results = [
                    {key: str(value) if value is not None else None 
                     for key, value in row._asdict().items()}
                    for row in final_state["execution_result"]
                    if hasattr(row, '_asdict')
                ]
            summary = f"Query executed successfully after {final_state['iteration_count']} iterations. Found {len(formatted_results)} rows."
        except Exception as e:
            log_agent_step("System", "Error formatting results", {"error": str(e)}, level="ERROR")
            formatted_results = []
            summary = f"Query executed with {len(final_state['execution_result'])} results, but there was an error formatting them."
    elif final_state["error"]:
        summary = f"Query failed after {final_state['iteration_count']} iterations (max {max_iterations}). Final error: {final_state['error']}\nFinal SQL attempted: {final_state['generated_query']}"
    else:
        summary = f"No results found after {final_state['iteration_count']} iterations (may have been revised).\nSQL: {final_state['generated_query']}"

    log_agent_step("System", "Query processing completed", {
        "run_id": run_id,
        "final_state": {
            "has_error": bool(final_state.get("error")),
            "is_empty_result": final_state.get("is_empty_result", False),
            "iterations": final_state.get("iteration_count", 0),
            "final_query": final_state.get("generated_query", ""),
            "result_rows": len(final_state.get("execution_result", [])),
            "error": final_state.get("error", "None")
        }
    })
    
    return {
        "summary": summary,
        "results": formatted_results,
        "sql_query": final_state.get("generated_query", ""),
        "error": final_state.get("error"),
        "selected_tables": final_state.get("selected_tables", []),
        "iteration_count": final_state.get("iteration_count", 0),
        "is_empty_result": final_state.get("is_empty_result", True)
    }

if __name__ == "__main__":
    result = run_conversational_query("[CompanyID: 1] Show me all upcoming schedules for this week.")
    print(result["summary"])
    if result["results"]:
        print("Full results:", result["results"])



def get_report_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """
You are a reporting engine. 
Given the user's request and database schema, generate:

1. A T-SQL query with parameter placeholders like @FromDate, @CustomerId, @Status etc.
2. A JSON array of filters detected from the user query.

RULES FOR FILTERS:
- Every date range → produce "From Date" and "To Date" filters.
- Any entity (customer, employee, project, user) → type = "master"
- Every filter must include:
    - name
    - placeholder
    - type (datetime, number, text, master)
    - isRequired (true/false)
    - metadata if master → include API source

Return output ONLY in this JSON structure:
{
  "query": "<sql>",
  "filters": [ ... ]
}
"""),
        ("human", "{user_query}")
    ])

def report_generator(state: AgentState) -> AgentState:
    try:
        prompt = get_report_generation_prompt()
        chain = prompt | llm | JsonOutputParser()

        response = chain.invoke({
            "user_query": state["user_query"],
        })

        # Save separately from SQL generator
        state["generated_report"] = response
        state["generated_query"] = response.get("query", "")
        state["filters"] = response.get("filters", [])

        state["messages"].append(AIMessage(content="Report generated successfully."))

        return state

    except Exception as e:
        state["error"] = str(e)
        return state
