import logging
import json
import os
from logging.handlers import TimedRotatingFileHandler

class JsonFormatter(logging.Formatter):
    """
    Formats log records as a JSON string.
    """
    def format(self, record):
        base = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "funcName": record.funcName,
            "lineNo": record.lineno,
            "logger": record.name,
        }
        # If message is a dict, merge it; else use as string message
        msg = record.msg
        if isinstance(msg, dict):
            log_record = {**base, **msg}
        else:
            log_record = {**base, "message": record.getMessage()}
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

    # Dedicated audit logger -> security_events.log
    audit_logger = logging.getLogger("audit")
    audit_logger.setLevel(log_level)
    audit_handler = TimedRotatingFileHandler("security_events.log", when="midnight", interval=1, backupCount=14)
    audit_handler.setFormatter(formatter)
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level {log_level}.")
