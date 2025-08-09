from typing import Dict, Any

def validate_rep_completeness(rep_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates if the output from a REP protocol is complete.
    A complete REP output should have both premises and a conclusion.
    """
    premises = rep_output.get("premises", [])
    conclusion = rep_output.get("conclusion")

    if premises and isinstance(premises, list) and len(premises) > 0 and conclusion:
        return {"is_complete": True, "score": 1.0, "reason": "Output contains premises and a conclusion."}

    missing = []
    if not premises: missing.append("premises")
    if not conclusion: missing.append("conclusion")

    return {"is_complete": False, "score": 0.2, "reason": f"Output is incomplete. Missing fields: {', '.join(missing)}."}


def validate_esl_completeness(esl_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates if the output from an ESL protocol is complete.
    A complete ESL output should have a valence and detected emotions.
    """
    valence = esl_output.get("valence")
    emotions = esl_output.get("detected_emotions") # Can be an empty list for neutral

    if valence and emotions is not None:
        return {"is_complete": True, "score": 1.0, "reason": "Output contains valence and emotion data."}

    return {"is_complete": False, "score": 0.2, "reason": "Output is missing 'valence' or 'detected_emotions'."}


def validate_mtp_completeness(mtp_output: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """
    Validates if the output from an MTP protocol is complete.
    For long inputs, it's expected to extract at least one entity.
    """
    entities = mtp_output.get("extracted_entities", [])
    word_count = len(user_input.split())

    if word_count < 10 or entities:
        return {"is_complete": True, "score": 1.0, "reason": "MTP output is considered complete."}

    if word_count >= 10 and not entities:
        return {
            "is_complete": False,
            "score": 0.4,
            "reason": "Output may be incomplete; no entities were extracted from a text with significant length."
        }

    return {"is_complete": True, "score": 1.0, "reason": "MTP output is considered complete."}


if __name__ == '__main__':
    # Example Usage
    print("--- Testing Completeness Validators ---")

    # REP
    rep_complete = {"premises": ["A is B"], "conclusion": "C"}
    rep_incomplete = {"premises": ["A is B"]}
    print(f"REP Complete: {validate_rep_completeness(rep_complete)}")
    assert validate_rep_completeness(rep_complete)['is_complete'] is True
    print(f"REP Incomplete: {validate_rep_completeness(rep_incomplete)}")
    assert validate_rep_completeness(rep_incomplete)['is_complete'] is False

    # ESL
    esl_complete = {"valence": "positive", "detected_emotions": []}
    esl_incomplete = {"valence": "positive"}
    print(f"\nESL Complete: {validate_esl_completeness(esl_complete)}")
    assert validate_esl_completeness(esl_complete)['is_complete'] is True
    print(f"ESL Incomplete: {validate_esl_completeness(esl_incomplete)}")
    assert validate_esl_completeness(esl_incomplete)['is_complete'] is False

    # MTP
    mtp_complete1 = {"extracted_entities": [{"entity": "test"}]}
    mtp_complete2 = {"extracted_entities": []}
    mtp_incomplete = {"extracted_entities": []}
    long_input = "This is a very long sentence designed to test the completeness validator for MTP."
    short_input = "Short."
    print(f"\nMTP Complete (found entity): {validate_mtp_completeness(mtp_complete1, long_input)}")
    assert validate_mtp_completeness(mtp_complete1, long_input)['is_complete'] is True
    print(f"MTP Complete (short input): {validate_mtp_completeness(mtp_complete2, short_input)}")
    assert validate_mtp_completeness(mtp_complete2, short_input)['is_complete'] is True
    print(f"MTP Incomplete (long input, no entity): {validate_mtp_completeness(mtp_incomplete, long_input)}")
    assert validate_mtp_completeness(mtp_incomplete, long_input)['is_complete'] is False
