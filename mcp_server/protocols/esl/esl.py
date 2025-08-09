import logging
import re
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class ESL:
    """
    A sophisticated rule-based Emotional State Layer (ESL) using regex and
    contextual analysis to perform multi-dimensional emotion detection.
    This implementation does not require any ML libraries.
    """

    def __init__(self):
        self.emotion_patterns = {
            # Primary Emotions
            "joy": {"pattern": r"\b(thrilled|elated|gleeful|happy|joyful|pleased|excited|delighted|wonderful|amazing|great|fantastic)\b", "dimension": "primary", "valence": "positive"},
            "sadness": {"pattern": r"\b(sad|unhappy|sorrowful|dejected|miserable|disappointed|heartbroken|grief|depressed|terrible)\b", "dimension": "primary", "valence": "negative"},
            "anger": {"pattern": r"\b(angry|furious|enraged|irate|livid|outraged|frustrated|annoyed|pissed off)\b", "dimension": "primary", "valence": "negative"},
            "fear": {"pattern": r"\b(fearful|scared|terrified|anxious|afraid|panicked|worried)\b", "dimension": "primary", "valence": "negative"},
            "surprise": {"pattern": r"\b(surprised|astonished|amazed|shocked|startled)\b", "dimension": "primary", "valence": "neutral"},
            "disgust": {"pattern": r"\b(disgusted|disgusting|repulsed|revolted|sickened|appalled)\b", "dimension": "primary", "valence": "negative"},
            # Social Emotions
            "gratitude": {"pattern": r"\b(grateful|thankful|appreciate|thanks|thank you)\b", "dimension": "social", "valence": "positive"},
            "empathy": {"pattern": r"\b(empathetic|sympathetic|understand your feeling)\b", "dimension": "social", "valence": "positive"},
            "pride": {"pattern": r"\b(proud)\b", "dimension": "social", "valence": "positive"},
            "guilt": {"pattern": r"\b(guilty|ashamed|regretful)\b", "dimension": "social", "valence": "negative"},
            "envy": {"pattern": r"\b(envious|jealous)\b", "dimension": "social", "valence": "negative"},
            # Cognitive Emotions
            "hope": {"pattern": r"\b(hopeful|optimistic)\b", "dimension": "cognitive", "valence": "positive"},
            "curiosity": {"pattern": r"\b(curious|intrigued|inquisitive|wondering)\b", "dimension": "cognitive", "valence": "neutral"},
            "confusion": {"pattern": r"\b(confused|puzzled|baffled|bewildered|don't understand)\b", "dimension": "cognitive", "valence": "negative"},
            "confidence": {"pattern": r"\b(confident|certain|sure|believe)\b", "dimension": "cognitive", "valence": "positive"},
            "doubt": {"pattern": r"\b(doubtful|uncertain|unsure|skeptical)\b", "dimension": "cognitive", "valence": "negative"},
        }

        self.modifiers = {
            "intensifiers": re.compile(r"\b(absolutely|extremely|incredibly|really|so|very|completely|totally)\b"),
            "diminishers": re.compile(r"\b(slightly|somewhat|a bit|a little|partially)\b"),
            "negations": re.compile(r"\b(not|never|no|n't|ain't|hardly|without)\b"),
            "uncertainty": re.compile(r"\b(maybe|perhaps|might be|could be|i guess|i suppose)\b")
        }

        self.temporal_indicators = re.compile(r"\b(always|never|sometimes|often|rarely|usually)\b")

    def _analyze_context(self, text: str, match_start: int, window=40) -> Dict[str, Any]:
        """Analyzes the text preceding an emotion keyword for modifiers. Increased window to 40."""
        context_text = text[max(0, match_start - window):match_start]
        return {
            "intensified": bool(self.modifiers["intensifiers"].search(context_text)),
            "diminished": bool(self.modifiers["diminishers"].search(context_text)),
            "negated": bool(self.modifiers["negations"].search(context_text)),
            "uncertain": bool(self.modifiers["uncertainty"].search(context_text)),
        }

    def _calculate_scores(self, context_mods: Dict[str, Any]) -> Tuple[float, float]:
        intensity = 0.6
        confidence = 0.9

        if context_mods["intensified"]: intensity = min(1.0, intensity + 0.3)
        if context_mods["diminished"]: intensity = max(0.1, intensity - 0.3)
        if context_mods["negated"]: intensity = 0.5
        if context_mods["uncertain"]: confidence = max(0.3, confidence - 0.4)

        return round(intensity, 2), round(confidence, 2)

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        user_input = data.get("user_input", "")
        if not user_input:
            return {"status": "error", "reason": "No user_input provided."}

        lower_input = user_input.lower()

        aggregated_emotions = {}

        for emotion, details in self.emotion_patterns.items():
            for match in re.finditer(details["pattern"], lower_input):
                context_mods = self._analyze_context(lower_input, match.start())
                intensity, confidence = self._calculate_scores(context_mods)

                valence = details["valence"]
                if context_mods["negated"]:
                    if valence == "positive": valence = "negative"
                    elif valence == "negative": valence = "positive"

                if emotion not in aggregated_emotions:
                    aggregated_emotions[emotion] = {
                        "scores": [], "confidences": [], "valences": [], "dimension": details["dimension"]
                    }

                aggregated_emotions[emotion]["scores"].append(intensity)
                aggregated_emotions[emotion]["confidences"].append(confidence)
                aggregated_emotions[emotion]["valences"].append(valence)

        is_question = "?" in user_input

        if not aggregated_emotions:
            return {
                "emotional_state": "neutral", "valence": "neutral", "intensity": 0, "salience": 0, "confidence": 1.0,
                "detected_emotions": [], "contextual_factors": {"is_question": is_question},
                "explanation": "No significant emotional content detected.",
                "analysis_type": "linguistic_patterns", "ml_ready": True
            }

        detected_emotions = []
        for emotion, data in aggregated_emotions.items():
            avg_intensity = round(sum(data["scores"]) / len(data["scores"]), 2)
            avg_confidence = round(sum(data["confidences"]) / len(data["confidences"]), 2)
            final_valence = max(set(data["valences"]), key=data["valences"].count)

            detected_emotions.append({
                "emotion": emotion, "intensity": avg_intensity, "confidence": avg_confidence,
                "dimension": data["dimension"], "valence": final_valence
            })

        dominant_emotion = max(detected_emotions, key=lambda x: x["intensity"])
        overall_intensity = dominant_emotion["intensity"]
        overall_confidence = sum(e["confidence"] for e in detected_emotions) / len(detected_emotions)

        valences = {e["valence"] for e in detected_emotions if e["valence"] in ["positive", "negative"]}
        if len(valences) > 1: overall_valence = "mixed"
        else: overall_valence = dominant_emotion["valence"]

        salience = min(1.0, (len(detected_emotions) * 0.2) + (overall_intensity * 0.5))

        contextual_factors = {
            "is_question": is_question,
            "has_negation": bool(self.modifiers["negations"].search(lower_input)),
            "temporal_indicators": list(set(self.temporal_indicators.findall(lower_input)))
        }

        other_emotions_list = [e['emotion'] for e in detected_emotions if e['emotion'] != dominant_emotion['emotion']]
        explanation = (f"Detected {dominant_emotion['emotion']} as the dominant emotion with {overall_valence} valence. "
                       f"Overall intensity is {overall_intensity} with a confidence of {round(overall_confidence, 2)}. ")
        if other_emotions_list:
            explanation += f"Other emotions detected: {', '.join(other_emotions_list)}. "
        if contextual_factors["has_negation"]:
            explanation += "Negation was detected, which may have inverted emotional valence. "

        return {
            "emotional_state": dominant_emotion["emotion"], "valence": overall_valence, "intensity": overall_intensity,
            "salience": round(salience, 2), "confidence": round(overall_confidence, 2),
            "detected_emotions": sorted(detected_emotions, key=lambda x: x["intensity"], reverse=True),
            "contextual_factors": contextual_factors, "explanation": explanation,
            "analysis_type": "linguistic_patterns", "ml_ready": True
        }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    esl = ESL()
    print("--- Testing Sophisticated Rule-Based ESL Protocol (v3) ---")

    tests = [
        "I'm absolutely thrilled about this amazing opportunity!",
        "I'm not happy about this situation.",
        "I'm slightly confused and don't understand.",
        "Thank you so much, I really appreciate it.",
        "I am so incredibly angry and frustrated, but I guess I'm also a little hopeful.",
        "This is a terrible, disgusting, and very bad situation.",
        "Do you think this is a good idea? I'm a bit unsure."
    ]

    for i, text in enumerate(tests):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input: '{text}'")
        result = esl.execute({"user_input": text})
        import json
        print(f"Output: {json.dumps(result, indent=2)}")
