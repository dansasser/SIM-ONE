import re
import statistics
from typing import Dict, Any, List, Set

# --- Dictionaries for Rule-Based Checks ---

SENTIMENT_KEYWORDS = {
    "positive": {"success", "successful", "good", "great", "happy", "love", "excellent", "achievement", "promotion"},
    "negative": {"failure", "bad", "terrible", "sad", "hate", "problem", "error", "frustrated", "weak", "stress", "anger"}
}

ENTITY_EMOTION_MAP = {
    "promotion": "positive", "success": "positive", "achievement": "positive",
    "problem": "negative", "failure": "negative", "stress": "negative"
}

TEMPORAL_KEYWORDS = {
    "past": {"yesterday", "was", "were", "last week", "ago"},
    "present": {"today", "is", "are", "now"},
    "future": {"tomorrow", "will", "next week"}
}

# --- Individual Checker Functions ---

def check_sentiment_alignment(rep_output: Dict, esl_output: Dict) -> float:
    """Compares REP conclusion sentiment with ESL valence."""
    rep_conclusion = rep_output.get("conclusion", "").lower()
    esl_valence = esl_output.get("valence", "neutral")

    if not rep_conclusion or esl_valence == "neutral":
        return 1.0

    rep_pos = any(word in rep_conclusion for word in SENTIMENT_KEYWORDS["positive"])
    rep_neg = any(word in rep_conclusion for word in SENTIMENT_KEYWORDS["negative"])

    if (rep_pos and not rep_neg and esl_valence == "positive") or \
       (rep_neg and not rep_pos and esl_valence == "negative"):
        return 1.0 # Perfect alignment
    if (rep_pos and not rep_neg and esl_valence == "negative") or \
       (rep_neg and not rep_pos and esl_valence == "positive"):
        return 0.1 # Clear contradiction
    if not rep_pos and not rep_neg and esl_valence != "neutral":
        return 0.2 # Neutral statement with strong emotion is a mismatch

    return 0.6 # For other ambiguous cases (e.g., mixed/mixed)

def check_entity_emotion_correlation(mtp_output: Dict, esl_output: Dict) -> float:
    """Checks if extracted entities align with the emotional context."""
    entities = [e.get('entity', '').lower() for e in mtp_output.get("extracted_entities", [])]
    esl_valence = esl_output.get("valence", "neutral")

    if not entities or esl_valence == "neutral":
        return 1.0

    scores = []
    for entity in entities:
        expected_valence = ENTITY_EMOTION_MAP.get(entity)
        if expected_valence:
            scores.append(1.0 if expected_valence == esl_valence else 0.0)

    return statistics.mean(scores) if scores else 1.0

def check_confidence_consistency(protocol_outputs: List[Dict]) -> float:
    """Calculates the consistency of confidence scores across protocols."""
    confidences = [p.get("confidence", 0.7) for p in protocol_outputs if "confidence" in p]
    if len(confidences) < 2:
        return 1.0

    # High standard deviation means low consistency. We want a score where 1.0 is best.
    std_dev = statistics.stdev(confidences)
    return max(0.0, 1.0 - (std_dev * 2)) # Scale stdev to be more impactful

def check_temporal_consistency(protocol_outputs: List[Dict]) -> float:
    """Checks for contradictory temporal keywords across all protocol outputs."""
    all_text = " ".join([str(v) for p in protocol_outputs for v in p.values()]).lower()

    found_tenses: Set[str] = set()
    for tense, keywords in TEMPORAL_KEYWORDS.items():
        if any(word in all_text for word in keywords):
            found_tenses.add(tense)

    # If more than one tense is explicitly found, it's a potential contradiction.
    return 0.2 if len(found_tenses) > 1 else 1.0

# --- Main Orchestrator Function ---

def run_coherence_checks(rep_output: Dict, esl_output: Dict, mtp_output: Dict) -> Dict:
    """
    Orchestrates all coherence checks and returns a final report.
    """
    protocol_outputs = [rep_output, esl_output, mtp_output]
    issues = []

    # Run checks
    sentiment_score = check_sentiment_alignment(rep_output, esl_output)
    entity_emotion_score = check_entity_emotion_correlation(mtp_output, esl_output)
    confidence_score = check_confidence_consistency(protocol_outputs)
    temporal_score = check_temporal_consistency(protocol_outputs)

    alignment_scores = {
        "sentiment_alignment": sentiment_score,
        "entity_emotion_correlation": entity_emotion_score,
        "confidence_consistency": confidence_score,
        "temporal_consistency": temporal_score
    }

    # Determine issues and overall coherence based on a "weakest link" model
    is_coherent = True
    critical_threshold = 0.5 # Any score below this makes the whole thing incoherent

    if sentiment_score < critical_threshold:
        issues.append("Sentiment mismatch between reasoning and emotion.")
        is_coherent = False
    if entity_emotion_score < critical_threshold:
        issues.append("Entity-emotion mismatch detected.")
        is_coherent = False
    if confidence_score < critical_threshold:
        issues.append("Confidence scores are highly inconsistent.")
        is_coherent = False
    if temporal_score < critical_threshold:
        issues.append("Temporal contradiction detected.")
        is_coherent = False

    # The overall score is the average, but if incoherent, it's penalized to be the minimum score.
    overall_score = statistics.mean(alignment_scores.values())
    if not is_coherent:
        overall_score = min(alignment_scores.values())

    return {
        "coherence_score": round(overall_score, 2),
        "is_coherent": is_coherent,
        "issues_detected": issues,
        "alignment_scores": {k: round(v, 2) for k, v in alignment_scores.items()},
        "recommendations": ["Review flagged issues for deeper analysis."] if issues else []
    }

if __name__ == '__main__':
    print("--- Testing Real Coherence Validation Algorithm ---")

    # Test Case 1 - COHERENT
    rep1 = {"conclusion": "Project successful", "confidence": 0.9}
    esl1 = {"valence": "positive", "confidence": 0.8}
    mtp1 = {"extracted_entities": [{"entity": "promotion"}, {"entity": "achievement"}]}

    result1 = run_coherence_checks(rep1, esl1, mtp1)
    print("\n--- Test Case 1 (COHERENT) ---")
    import json
    print(json.dumps(result1, indent=2))
    assert result1['is_coherent'] is True
    assert result1['coherence_score'] > 0.8

    # Test Case 2 - INCOHERENT
    rep2 = {"conclusion": "Everything is fine", "confidence": 0.9} # Neutral conclusion
    esl2 = {"valence": "negative", "confidence": 0.8} # Negative emotion
    mtp2 = {"extracted_entities": [{"entity": "problem"}, {"entity": "failure"}]} # Negative entities

    result2 = run_coherence_checks(rep2, esl2, mtp2)
    print("\n--- Test Case 2 (INCOHERENT) ---")
    print(json.dumps(result2, indent=2))
    assert result2['is_coherent'] is False
    assert result2['coherence_score'] < 0.5
    assert len(result2['issues_detected']) > 0

    print("\nAll tests passed!")
