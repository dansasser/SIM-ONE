#!/usr/bin/env python3
"""
Tool: Validation and Verification Protocol (VVP)
Input: Rules and data to validate
Output: Validation status and verification results

MCP_TOOL_ENTRYPOINT

The VVP protocol provides validation and verification capabilities for:
- Input data validation
- Rule structure verification
- Logical consistency checking
- Format compliance assessment

## Usage

### Validate rules:
```bash
python run_vvp_tool.py --json '{"rules": [[["a"], "b"], [["b"], "c"]]}'
```

### From file:
```bash
python run_vvp_tool.py --file validation_input.json
```

### From stdin:
```bash
echo '{"rules": [[["premise"], "conclusion"]]}' | python run_vvp_tool.py
```

## Input Format (JSON)

```json
{
  "rules": [
    [["premise1", "premise2"], "conclusion"]
  ]
}
```

## Output Format

```json
{
  "validation_status": "success",
  "reason": "Input validation passed: 2 rules were checked and appear well-formed."
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
    from mcp_server.protocols.vvp.vvp import VVP
except ImportError as e:
    print(json.dumps({"error": f"Failed to import VVP: {e}"}), file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="VVP - Validation and Verification Protocol")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--file', '-f', help='JSON file with validation input')
    input_group.add_argument('--json', '-j', help='JSON string with validation input')

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
            parser.print_help()
            sys.exit(1)

        if not data:
            print(json.dumps({"error": "No input provided"}), file=sys.stderr)
            sys.exit(1)

        # Execute VVP
        vvp = VVP()
        result = vvp.execute(data)

        print(json.dumps(result, indent=2))

        # Exit code based on validation status
        if result.get("validation_status") == "success":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
