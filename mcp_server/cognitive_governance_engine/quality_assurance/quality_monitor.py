from collections import defaultdict, deque
from typing import Dict, Deque
import statistics

class QualityMonitor:
    """
    Tracks quality scores for each protocol over time to detect trends.
    This is a simple in-memory implementation.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern to ensure one monitor across the system
        if cls._instance is None:
            cls._instance = super(QualityMonitor, cls).__new__(cls)
            # Use a defaultdict with a deque of max length 10
            cls._instance.history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=10))
        return cls._instance

    def add_score(self, protocol_name: str, score: float):
        """Adds a new quality score to the history for a given protocol."""
        if not (0.0 <= score <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0")
        self.history[protocol_name].append(score)

    def get_trend(self, protocol_name: str) -> Dict[str, any]:
        """
        Analyzes the recent history of scores to determine a trend.

        Returns:
            A dictionary containing the trend ('stable', 'improving', 'declining')
            and the average score.
        """
        scores = self.history.get(protocol_name)
        if not scores or len(scores) < 4:
            # Not enough data to determine a meaningful trend
            avg_score = statistics.mean(scores) if scores else 0.0
            return {"trend": "stable", "average_score": round(avg_score, 2), "reason": "Not enough data for trend analysis."}

        avg_score = statistics.mean(scores)

        # Split scores into two halves for trend comparison
        mid_point = len(scores) // 2
        first_half_avg = statistics.mean(list(scores)[:mid_point])
        second_half_avg = statistics.mean(list(scores)[mid_point:])

        # Define a threshold for a significant change
        change_threshold = 0.15

        if second_half_avg > first_half_avg + change_threshold:
            trend = "improving"
        elif second_half_avg < first_half_avg - change_threshold:
            trend = "declining"
        else:
            trend = "stable"

        return {"trend": trend, "average_score": round(avg_score, 2)}

    def get_all_history(self) -> Dict[str, list]:
        """Returns the entire score history."""
        # Convert deques to lists for easier JSON serialization if needed
        return {protocol: list(scores) for protocol, scores in self.history.items()}

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Quality Monitor ---")
    monitor = QualityMonitor()

    # Test 1: Stable trend
    for score in [0.8, 0.85, 0.8, 0.82]:
        monitor.add_score("REP", score)
    trend1 = monitor.get_trend("REP")
    print(f"Trend 1 (Stable): {trend1}")
    assert trend1['trend'] == 'stable'

    # Test 2: Declining trend
    for score in [0.9, 0.85, 0.6, 0.55, 0.5]:
        monitor.add_score("ESL", score)
    trend2 = monitor.get_trend("ESL")
    print(f"Trend 2 (Declining): {trend2}")
    assert trend2['trend'] == 'declining'

    # Test 3: Improving trend
    for score in [0.5, 0.6, 0.8, 0.85, 0.9]:
        monitor.add_score("MTP", score)
    trend3 = monitor.get_trend("MTP")
    print(f"Trend 3 (Improving): {trend3}")
    assert trend3['trend'] == 'improving'

    # Verify singleton behavior
    monitor2 = QualityMonitor()
    print(f"\nESL history from second monitor instance: {monitor2.get_all_history()['ESL']}")
    assert monitor is monitor2
    assert len(monitor2.get_all_history()['ESL']) == 5
    print("Singleton test passed.")
