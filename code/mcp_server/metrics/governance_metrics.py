from typing import Dict

_counters: Dict[str, int] = {
    "governance_coherence_failures": 0,
    "governance_aborts": 0,
    "recovery_retries": 0,
    "recovery_fallbacks": 0,
    "openai_calls": 0,
    "openai_errors": 0,
    "json_retries": 0,
}

# Optional Prometheus support
_PROM: Dict[str, object] = {}
try:
    from prometheus_client import Counter  # type: ignore

    _PROM = {
        "governance_coherence_failures": Counter(
            "simone_governance_coherence_failures_total",
            "Total number of coherence failures flagged by governance.",
        ),
        "governance_aborts": Counter(
            "simone_governance_aborts_total",
            "Total number of governance-enforced aborts.",
        ),
        "recovery_retries": Counter(
            "simone_recovery_retries_total",
            "Total number of recovery retries attempted.",
        ),
        "recovery_fallbacks": Counter(
            "simone_recovery_fallbacks_total",
            "Total number of fallbacks used by recovery.",
        ),
        "openai_calls": Counter(
            "simone_openai_calls_total",
            "Total number of OpenAI API calls.",
        ),
        "openai_errors": Counter(
            "simone_openai_errors_total",
            "Total number of OpenAI API call errors.",
        ),
        "json_retries": Counter(
            "simone_json_retries_total",
            "Total number of JSON parse retries across engines.",
        ),
    }
except Exception:
    _PROM = {}


def inc(name: str, delta: int = 1):
    if name in _counters:
        _counters[name] += delta
    else:
        _counters[name] = delta
    c = _PROM.get(name)
    if c:
        # Increment one-by-one for clarity; delta > 1 loops
        for _ in range(delta):
            try:
                c.inc()  # type: ignore[attr-defined]
            except Exception:
                pass


def snapshot() -> Dict[str, int]:
    return dict(_counters)
