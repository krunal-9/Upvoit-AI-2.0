import logging
import os
import json
from datetime import datetime
from typing import Any, Dict, Optional, Union

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Create a formatter with colors for console
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

# JSON Formatter for Audit Logs
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add structured data if available
        if hasattr(record, "data"):
            log_record["data"] = record.data
            
        # Add exception info if available
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

# Configure root logger
def configure_logging():
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"audit_{timestamp}.json")
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure file handler with JSON formatter
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO) # Audit logs usually need INFO level
    file_handler.setFormatter(JSONFormatter())
    
    # Configure console handler with Color formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)

    # Suppress noisy loggers
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("pymongo").setLevel(logging.WARNING)
    logging.getLogger("pymongo.connection").setLevel(logging.WARNING)
    logging.getLogger("pymongo.pool").setLevel(logging.WARNING)
    logging.getLogger("pymongo.server").setLevel(logging.WARNING)
    
    return log_file


logger = logging.getLogger(__name__)

def _format_data(data: Any, indent: int = 2) -> str:
    """Format data for logging, handling various types."""
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
    """
    Log a step in the agent's execution with structured data.
    
    Args:
        agent_name: Name of the agent or component
        message: Human-readable message
        data: Optional dictionary of structured data
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create a logger for the specific agent
    agent_logger = logging.getLogger(agent_name)
    
    # Pass the data dict as an extra field so JSONFormatter can pick it up
    extra = {"data": data} if data else {}
    
    agent_logger.log(log_level, message, extra=extra)

def log_error(
    agent_name: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR"
) -> None:
    """
    Log an error with context.
    
    Args:
        agent_name: Name of the agent or component
        error: The exception that was raised
        context: Additional context about the error
        level: Log level (default: ERROR)
    """
    error_data = {
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        **({} if context is None else context)
    }
    log_agent_step(agent_name, "Error occurred", error_data, level)
