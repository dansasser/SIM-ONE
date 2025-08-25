import logging
import json
import os
from logging.handlers import TimedRotatingFileHandler

class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    """
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
        }
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_record)

def setup_logging():
    """
    Sets up structured JSON logging for the application.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    formatter = JsonFormatter()

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler (rotates every day, keeps 7 days of backups)
    file_handler = TimedRotatingFileHandler("mcp_server.log", when="midnight", interval=1, backupCount=7)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {log_level}.")
