import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def assess_recovery_performance(resilience_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates assessing the performance of error recovery strategies.

    Args:
        resilience_stats: Statistics from the ResilienceMonitor.

    Returns:
        A dictionary with assessments of the recovery strategies.
    """
    assessment = {
        "best_strategy_per_protocol": {},
        "worst_strategy_per_protocol": {}
    }

    for protocol, strategies in resilience_stats.items():
        best_rate = -1.0
        worst_rate = 2.0
        best_strategy = None
        worst_strategy = None

        for strategy, stats in strategies.items():
            attempts = stats.get('attempts', 0)
            successes = stats.get('successes', 0)
            if attempts == 0:
                continue

            success_rate = successes / attempts

            if success_rate > best_rate:
                best_rate = success_rate
                best_strategy = strategy

            if success_rate < worst_rate:
                worst_rate = success_rate
                worst_strategy = strategy

        if best_strategy:
            assessment["best_strategy_per_protocol"][protocol] = {"strategy": best_strategy, "rate": best_rate}
        if worst_strategy:
             assessment["worst_strategy_per_protocol"][protocol] = {"strategy": worst_strategy, "rate": worst_rate}

    logger.info(f"Recovery performance assessment complete: {assessment}")
    return assessment

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Performance Assessor ---")
    mock_stats = {
        "REP": {
            "retry": {"attempts": 10, "successes": 8}, # 80%
            "use_fallback": {"attempts": 2, "successes": 2} # 100%
        },
        "ESL": {
            "retry": {"attempts": 5, "successes": 1} # 20%
        }
    }

    assessment = assess_recovery_performance(mock_stats)
    print(f"Assessment result: {assessment}")

    assert assessment['best_strategy_per_protocol']['REP']['strategy'] == 'use_fallback'
    assert assessment['worst_strategy_per_protocol']['REP']['strategy'] == 'retry'
    assert assessment['best_strategy_per_protocol']['ESL']['strategy'] == 'retry' # Also the worst
    print("Performance assessor tests passed.")
