from typing import Dict, Any

class FallbackManager:
    """
    Provides default, safe fallback data for protocols that have failed.
    """

    def get_fallback(self, protocol_name: str) -> Dict[str, Any]:
        """
        Returns the appropriate fallback output for a given protocol.
        """
        if protocol_name == "REP":
            return self.get_fallback_rep()
        elif protocol_name == "ESL":
            return self.get_fallback_esl()
        elif protocol_name == "MTP":
            return self.get_fallback_mtp()
        else:
            return {
                "status": "fallback",
                "error": f"No fallback defined for protocol '{protocol_name}'."
            }

    def get_fallback_rep(self) -> Dict[str, Any]:
        """Returns a neutral, generic reasoning output."""
        return {
            "status": "fallback",
            "conclusion": "Unable to determine a conclusion due to a preceding error.",
            "premises": [],
            "confidence": 0.1,
            "reasoning_type": "none"
        }

    def get_fallback_esl(self) -> Dict[str, Any]:
        """Returns a neutral emotional output."""
        return {
            "status": "fallback",
            "emotional_state": "unknown",
            "valence": "neutral",
            "intensity": 0.0,
            "detected_emotions": []
        }

    def get_fallback_mtp(self) -> Dict[str, Any]:
        """Returns an empty entity list."""
        return {
            "status": "fallback",
            "extracted_entities": [],
            "entity_relationships": [],
            "explanation": "Entity extraction failed."
        }

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Fallback Manager ---")
    manager = FallbackManager()

    rep_fallback = manager.get_fallback("REP")
    print(f"REP Fallback: {rep_fallback}")
    assert rep_fallback['status'] == 'fallback'
    assert rep_fallback['conclusion'] is not None

    esl_fallback = manager.get_fallback("ESL")
    print(f"ESL Fallback: {esl_fallback}")
    assert esl_fallback['valence'] == 'neutral'

    mtp_fallback = manager.get_fallback("MTP")
    print(f"MTP Fallback: {mtp_fallback}")
    assert mtp_fallback['extracted_entities'] == []

    unknown_fallback = manager.get_fallback("UNKNOWN")
    print(f"Unknown Fallback: {unknown_fallback}")
    assert "No fallback defined" in unknown_fallback['error']

    print("\nAll fallback tests passed.")
