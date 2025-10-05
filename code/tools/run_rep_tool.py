#!/usr/bin/env python3
"""
Tool: Reasoning and Explanation Protocol (REP)
Input: Facts, rules, reasoning type
Output: Reasoning chain, conclusions, confidence scores

MCP_TOOL_ENTRYPOINT

The REP protocol provides advanced reasoning capabilities including:
- Deductive reasoning (modus ponens, modus tollens)
- Inductive reasoning (pattern recognition, generalization)
- Abductive reasoning (best explanation inference)
- Analogical reasoning (similarity-based inference)
- Causal reasoning (cause-effect relationships)

## Usage

### Deductive Reasoning:
```bash
python run_rep_tool.py --reasoning-type deductive \
  --facts "Socrates is a man" "All men are mortal" \
  --rules '[["Socrates is a man", "All men are mortal"], "Socrates is mortal"]'
```

### From JSON file:
```bash
python run_rep_tool.py --file reasoning_input.json
```

### From stdin:
```bash
echo '{"reasoning_type": "deductive", "facts": [...], "rules": [...]}' | python run_rep_tool.py
```

## Input Format (JSON)

```json
{
  "reasoning_type": "deductive",
  "facts": ["fact1", "fact2"],
  "rules": [
    [["premise1", "premise2"], "conclusion"]
  ],
  "context": "optional context"
}
```

## Output Format

```json
{
  "reasoning_type": "deductive",
  "conclusions": ["conclusion1", "conclusion2"],
  "confidence_scores": {"conclusion1": 0.95},
  "reasoning_chain": [...],
  "validation": {...},
  "explanation": "Step-by-step reasoning explanation"
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
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mcp_server.protocols.rep.rep import AdvancedREP
except ImportError as e:
    print(json.dumps({"error": f"Failed to import REP: {e}"}), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="REP - Reasoning and Explanation Protocol")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--file', '-f', help='JSON file with reasoning input')
    input_group.add_argument('--json', '-j', help='JSON string with reasoning input')

    parser.add_argument('--reasoning-type', choices=['deductive', 'inductive', 'abductive', 'analogical', 'causal'])
    parser.add_argument('--facts', nargs='+', help='List of facts')
    parser.add_argument('--rules', help='JSON string of rules')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)

    try:
        # Get input data
        if args.file:
            with open(args.file, 'r') as f:
                data = json.load(f)
        elif args.json:
            data = json.loads(args.json)
        elif not sys.stdin.isatty():
            data = json.load(sys.stdin)
        else:
            # Build from arguments
            data = {}
            if args.reasoning_type:
                data['reasoning_type'] = args.reasoning_type
            if args.facts:
                data['facts'] = args.facts
            if args.rules:
                data['rules'] = json.loads(args.rules)

        if not data:
            parser.print_help()
            sys.exit(1)

        # Execute REP
        rep = AdvancedREP()
        result = rep.execute(data)

        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
