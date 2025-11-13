import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

LOG_DIR = "logs"
MAX_LOG_SIZE_MB = 5
MAX_LOG_SIZE = MAX_LOG_SIZE_MB * 1024 * 1024

def get_daily_log_file() -> str:
    """Returns today's log file path."""
    date_str = datetime.now().strftime("%Y%m%d")
    base_file = os.path.join(LOG_DIR, f"agent_{date_str}.log")

    # If file doesn't exist, use it
    if not os.path.exists(base_file):
        return base_file

    # If file exceeds size, rotate with incremental suffix
    index = 1
    rotated_file = base_file
    while os.path.exists(rotated_file) and os.path.getsize(rotated_file) > MAX_LOG_SIZE:
        rotated_file = os.path.join(LOG_DIR, f"agent_{date_str}_{index}.log")
        index += 1

    return rotated_file

# Create a formatter with colors
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

# Configure root logger
def configure_logging():
    # Create a timestamp for the log file
    log_file = get_daily_log_file()

    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-10s | %(name)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
    
    return log_file

# Initialize logging
log_file = configure_logging()
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
    
    # Format the log message with agent name and message
    log_msg = f"{agent_name}: {message}"
    
    # Add formatted data if provided
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

# Example usage
if __name__ == "__main__":
    log_agent_step("TestAgent", "This is an info message", {"key": "value"})
    try:
        1 / 0
    except Exception as e:
        log_error("TestAgent", e, {"additional": "context"})
