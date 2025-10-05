#!/usr/bin/env python3
"""
Tool: Emotional State Layer (ESL) Protocol
Input: User input text
Output: Emotional analysis with detected emotions, valence, intensity

MCP_TOOL_ENTRYPOINT

The ESL protocol provides sophisticated emotional intelligence analysis including:
- Multi-dimensional emotion detection (primary, social, cognitive)
- Valence analysis (positive, negative, neutral, mixed)
- Intensity and confidence scoring
- Contextual modifiers (intensifiers, diminishers, negations)
- Emotional salience calculation

## Usage

### Analyze text:
```bash
python run_esl_tool.py --text "I'm absolutely thrilled about this!"
```

### From stdin:
```bash
echo "I'm feeling confused and uncertain" | python run_esl_tool.py
```

### From file:
```bash
python run_esl_tool.py --file user_message.txt
```

## Output Format

```json
{
  "emotional_state": "joy",
  "valence": "positive",
  "intensity": 0.9,
  "salience": 0.85,
  "confidence": 0.92,
  "detected_emotions": [
    {
      "emotion": "joy",
      "intensity": 0.9,
      "confidence": 0.92,
      "dimension": "primary",
      "valence": "positive"
    }
  ],
  "contextual_factors": {
    "is_question": false,
    "has_negation": false,
    "temporal_indicators": []
  },
  "explanation": "Detected joy as the dominant emotion..."
}
```

---

**Part of the SIM-ONE Framework**
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp_server.protocols.esl.esl import ESL
except ImportError as e:
    print(json.dumps({"error": f"Failed to import ESL: {e}"}), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="ESL - Emotional State Layer Protocol")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', '-t', help='Text to analyze')
    input_group.add_argument('--file', '-f', help='File containing text to analyze')

    parser.add_argument('--format', choices=['json', 'summary'], default='json')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        # Get input text
        if args.text:
            user_input = args.text
        elif args.file:
            with open(args.file, 'r') as f:
                user_input = f.read().strip()
        elif not sys.stdin.isatty():
            user_input = sys.stdin.read().strip()
        else:
            parser.print_help()
            sys.exit(1)

        if not user_input:
            print(json.dumps({"error": "No input provided"}), file=sys.stderr)
            sys.exit(1)

        # Execute ESL
        esl = ESL()
        result = esl.execute({"user_input": user_input})

        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            # Summary format
            print(f"Emotional State: {result['emotional_state']}")
            print(f"Valence: {result['valence']}")
            print(f"Intensity: {result['intensity']}")
            print(f"Confidence: {result['confidence']}")
            if result.get('detected_emotions'):
                print(f"\nDetected Emotions:")
                for emotion in result['detected_emotions'][:3]:
                    print(f"  - {emotion['emotion']}: {emotion['intensity']:.2f}")

        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
