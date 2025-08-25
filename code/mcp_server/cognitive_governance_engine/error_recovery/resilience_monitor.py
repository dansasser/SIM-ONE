from collections import defaultdict
from typing import Dict, Any

class ResilienceMonitor:
    """
    A singleton class to monitor the usage and success of error recovery strategies.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResilienceMonitor, cls).__new__(cls)
            # Nested dictionary to store stats: {protocol: {strategy: {attempts: x, successes: y}}}
            cls._instance.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        return cls._instance

    def record_strategy(self, protocol_name: str, strategy: str, success: bool):
        """
        Records an attempt to use a recovery strategy and whether it was successful.

        Args:
            protocol_name: The name of the protocol that failed.
            strategy: The name of the recovery strategy used (e.g., 'retry', 'use_fallback').
            success: A boolean indicating if the subsequent operation succeeded.
        """
        self.stats[protocol_name][strategy]['attempts'] += 1
        if success:
            self.stats[protocol_name][strategy]['successes'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns the collected resilience statistics.
        """
        # Convert defaultdicts to regular dicts for cleaner output
        return {p: {s: dict(v) for s, v in sv.items()} for p, sv in self.stats.items()}

    def get_success_rate(self, protocol_name: str, strategy: str) -> float:
        """
        Calculates the success rate for a specific recovery strategy.
        """
        attempts = self.stats[protocol_name][strategy]['attempts']
        successes = self.stats[protocol_name][strategy]['successes']

        if attempts == 0:
            return 0.0

        return successes / attempts

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Resilience Monitor ---")
    monitor = ResilienceMonitor()

    # Simulate some recovery attempts
    monitor.record_strategy("REP", "retry", True)
    monitor.record_strategy("REP", "retry", False)
    monitor.record_strategy("REP", "retry", True)
    monitor.record_strategy("ESL", "use_fallback", True) # Fallbacks are always "successful"

    stats = monitor.get_stats()
    print(f"Collected Stats: {stats}")

    rep_retry_stats = stats.get("REP", {}).get("retry", {})
    assert rep_retry_stats.get("attempts") == 3
    assert rep_retry_stats.get("successes") == 2

    esl_fallback_stats = stats.get("ESL", {}).get("use_fallback", {})
    assert esl_fallback_stats.get("attempts") == 1

    rep_retry_rate = monitor.get_success_rate("REP", "retry")
    print(f"\nREP 'retry' success rate: {rep_retry_rate:.2f}")
    assert abs(rep_retry_rate - (2/3)) < 0.01

    # Test singleton behavior
    monitor2 = ResilienceMonitor()
    monitor2.record_strategy("REP", "retry", True)
    stats2 = monitor.get_stats()
    print(f"\nUpdated Stats from original instance: {stats2}")
    assert stats2["REP"]["retry"]["attempts"] == 4
    print("Singleton test passed.")
