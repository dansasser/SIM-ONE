from typing import Dict, Any

from . import error_classifier
from .fallback_manager import FallbackManager
from .resilience_monitor import ResilienceMonitor

class RecoveryStrategist:
    """
    Selects an appropriate error recovery strategy based on the type of error,
    the protocol that failed, and the current retry count.
    """

    def __init__(self, max_retries: int = 2):
        self.fallback_manager = FallbackManager()
        self.resilience_monitor = ResilienceMonitor()
        self.max_retries = max_retries

    def select_strategy(self, error: Exception, protocol_name: str, retry_count: int) -> Dict[str, Any]:
        """
        Analyzes an error and determines the best recovery strategy.

        Returns:
            A dictionary containing the chosen 'strategy' and any associated 'data'
            (e.g., fallback data).
        """
        classified_error = error_classifier.classify_error(error, protocol_name)
        error_type = classified_error['type']
        severity = classified_error['severity']

        # Strategy 1: Abort on critical, unrecoverable errors
        if severity == 'high' and error_type != 'ConnectionError':
            strategy = "abort"
            self.resilience_monitor.record_strategy(protocol_name, strategy, success=False)
            return {"strategy": strategy, "data": None, "reason": f"Aborting due to high-severity error: {error_type}"}

        # Strategy 2: Retry if possible
        if retry_count < self.max_retries:
            strategy = "retry"
            # We don't record the outcome here, the caller will do that on the next attempt
            return {"strategy": strategy, "data": None, "reason": f"Attempting retry {retry_count + 1} of {self.max_retries}."}

        # Strategy 3: Use Fallback if retries are exhausted
        strategy = "use_fallback"
        fallback_data = self.fallback_manager.get_fallback(protocol_name)
        # Using a fallback is considered a "successful" recovery from the system's perspective
        self.resilience_monitor.record_strategy(protocol_name, strategy, success=True)
        return {
            "strategy": strategy,
            "data": fallback_data,
            "reason": f"Max retries exceeded. Using fallback data for {protocol_name}."
        }

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Recovery Strategist ---")
    strategist = RecoveryStrategist(max_retries=1)

    # Test Case 1: A recoverable error, first attempt -> should retry
    error1 = ValueError("Temporary issue")
    strategy1 = strategist.select_strategy(error1, "REP", retry_count=0)
    print(f"Test 1 (Retry): {strategy1}")
    assert strategy1['strategy'] == 'retry'

    # Test Case 2: A recoverable error, but max retries exceeded -> should use fallback
    error2 = ValueError("Temporary issue again")
    strategy2 = strategist.select_strategy(error2, "ESL", retry_count=1)
    print(f"Test 2 (Use Fallback): {strategy2}")
    assert strategy2['strategy'] == 'use_fallback'
    assert strategy2['data']['status'] == 'fallback'
    assert strategy2['data']['valence'] == 'neutral'

    # Test Case 3: A critical, unrecoverable error -> should abort
    error3 = TypeError("Fatal error in protocol logic")
    strategy3 = strategist.select_strategy(error3, "MTP", retry_count=0)
    print(f"Test 3 (Abort): {strategy3}")
    assert strategy3['strategy'] == 'abort'

    # Check that the monitor was updated for fallback and abort
    monitor_stats = strategist.resilience_monitor.get_stats()
    print(f"\nResilience Monitor Stats: {monitor_stats}")
    assert monitor_stats["ESL"]["use_fallback"]["attempts"] == 1
    assert monitor_stats["MTP"]["abort"]["attempts"] == 1
    print("All strategy tests passed.")
