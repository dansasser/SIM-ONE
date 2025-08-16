import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def assess_self_performance(governance_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates self-monitoring by analyzing a governance report.
    It checks overall coherence and quality to create a self-assessment summary.

    Args:
        governance_report: The aggregated report from other governance modules.
                           Expected to have keys like 'coherence' and 'quality'.

    Returns:
        A dictionary summarizing the self-assessment.
    """
    coherence_report = governance_report.get("coherence", {})
    quality_reports = governance_report.get("quality", {})

    is_coherent = coherence_report.get("is_coherent", True)

    # Calculate average quality score across all assessed protocols
    total_quality = 0
    num_protocols = 0
    if quality_reports:
        for report in quality_reports.values():
            total_quality += report.get("quality_score", 0.0)
            num_protocols += 1
    average_quality = total_quality / num_protocols if num_protocols > 0 else 0.0

    # Determine overall performance state
    performance_state = "nominal"
    issues = []
    if not is_coherent:
        performance_state = "degraded"
        issues.extend(coherence_report.get("reasons", ["Coherence check failed."]))

    if average_quality < 0.6:
        performance_state = "degraded"
        issues.append(f"Average quality score ({average_quality:.2f}) is below threshold.")

    logger.info(f"Self-assessment complete. Performance state: {performance_state}.")

    return {
        "performance_state": performance_state,
        "average_quality_score": round(average_quality, 2),
        "is_coherent": is_coherent,
        "identified_issues": issues
    }

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Self Monitor ---")

    # Test Case 1: Good performance
    report1 = {
        "coherence": {"is_coherent": True},
        "quality": {
            "REP": {"quality_score": 0.9},
            "ESL": {"quality_score": 0.85}
        }
    }
    assessment1 = assess_self_performance(report1)
    print(f"Assessment 1 (Good): {assessment1}")
    assert assessment1['performance_state'] == 'nominal'
    assert assessment1['average_quality_score'] == 0.88

    # Test Case 2: Poor performance (incoherent)
    report2 = {
        "coherence": {"is_coherent": False, "reasons": ["Mismatch between REP and ESL."]},
        "quality": {
            "REP": {"quality_score": 0.9},
            "ESL": {"quality_score": 0.85}
        }
    }
    assessment2 = assess_self_performance(report2)
    print(f"Assessment 2 (Incoherent): {assessment2}")
    assert assessment2['performance_state'] == 'degraded'
    assert "Mismatch" in assessment2['identified_issues'][0]

    # Test Case 3: Poor performance (low quality)
    report3 = {
        "coherence": {"is_coherent": True},
        "quality": {
            "REP": {"quality_score": 0.4},
            "ESL": {"quality_score": 0.5}
        }
    }
    assessment3 = assess_self_performance(report3)
    print(f"Assessment 3 (Low Quality): {assessment3}")
    assert assessment3['performance_state'] == 'degraded'
    assert "quality score" in assessment3['identified_issues'][0]
