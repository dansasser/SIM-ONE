import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("security_logger")
logger.setLevel(logging.WARNING)

# Create a handler for logging to a file
file_handler = logging.FileHandler('security_events.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def sanitize_message(detail: str) -> str:
    """
    Sanitizes error messages to prevent information leakage.
    In a real-world scenario, you might want to allow certain safe characters
    or use a more sophisticated sanitization library.
    """
    # For now, we return a generic message for any non-HTTPException error.
    return "An internal server error occurred. Please contact support."

async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handles generic, unexpected exceptions.
    Logs the full error for debugging and returns a sanitized message to the client.
    """
    # Log the detailed, sensitive error information
    logger.error(f"Unhandled exception for request {request.method} {request.url}: {exc}", exc_info=True)

    # Return a generic, non-informative error message to the client
    return JSONResponse(
        status_code=500,
        content={"detail": sanitize_message(str(exc))},
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles FastAPI's HTTPException.
    These are "expected" errors (like 404, 403) and can often be returned to the client.
    We log them for security monitoring.
    """
    # Log security-relevant HTTP exceptions
    if 400 <= exc.status_code < 500:
        logger.warning(f"Client error for request {request.method} {request.url}: {exc.detail}")
    else:
        logger.error(f"Server error for request {request.method} {request.url}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )

def add_exception_handlers(app):
    """Adds the custom exception handlers to the FastAPI app."""
    app.add_exception_handler(Exception, generic_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
