import json
from typing import Any, Callable
from mcp_server.metrics import governance_metrics as govm


def ensure_json(engine_call: Callable[[], str], tighten_call: Callable[[], str] | None = None, default_factory: Callable[[], dict] | None = None) -> dict:
    """
    Calls the engine and ensures JSON is returned.
    - engine_call: function returning a string (first attempt)
    - tighten_call: optional function for a stricter retry
    - default_factory: optional supplier for a safe default JSON on failure
    """
    text = engine_call() or ""
    try:
        return json.loads(text)
    except Exception:
        govm.inc("json_retries")
        if tighten_call:
            text2 = tighten_call() or ""
            try:
                return json.loads(text2)
            except Exception:
                pass
    return default_factory() if default_factory else {}

