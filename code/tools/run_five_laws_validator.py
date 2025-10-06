#!/usr/bin/env python3
"""
Tool: Five Laws Cognitive Governance Validator
Input: Text response to validate, optional context
Output: Compliance assessment with scores and recommendations

MCP_TOOL_ENTRYPOINT

This tool validates any AI-generated content against SIM-ONE's Five Laws of Cognitive Governance:

1. **Architectural Intelligence** - Intelligence from coordination, not brute force
2. **Cognitive Governance** - Governed processes over unconstrained generation
3. **Truth Foundation** - Absolute truth principles over probabilistic drift
4. **Energy Stewardship** - Computational efficiency and resource awareness
5. **Deterministic Reliability** - Consistent, predictable outcomes

## Usage

### From command line with text argument:
```bash
python run_five_laws_validator.py --text "Response to validate"
```

### From stdin:
```bash
echo "AI response text" | python run_five_laws_validator.py
```

### From file:
```bash
python run_five_laws_validator.py --file response.txt
```

### With strictness level:
```bash
python run_five_laws_validator.py --text "..." --strictness strict
```

### With context:
```bash
python run_five_laws_validator.py --text "..." --context '{"domain": "scientific"}'
```

## Output Format

JSON structure:
```json
{
  "scores": {
    "law1_architectural_intelligence": 85.5,
    "law2_cognitive_governance": 72.3,
    "law3_truth_foundation": 90.1,
    "law4_energy_stewardship": 68.7,
    "law5_deterministic_reliability": 78.2,
    "overall_compliance": 79.8
  },
  "pass_fail_status": "PASS",
  "violations": ["List of specific violations"],
  "recommendations": ["Actionable recommendations"],
  "strengths": ["Identified strengths"],
  "detailed_results": {...}
}
```

## Use Cases

### Paper2Agent Self-Governance
Paper2Agent can validate its own responses before returning them to users:
```python
result = run_five_laws_validator(my_response)
if result["pass_fail_status"] == "PASS":
    return my_response
else:
    refine_response(result["recommendations"])
```

### Multi-Agent Validation
One agent validates another's output:
```bash
other_agent_output | python run_five_laws_validator.py
```

### Continuous Integration
Validate AI-generated content in CI/CD pipelines:
```yaml
- run: python code/tools/run_five_laws_validator.py --file ai_output.txt
```

---

**Part of the SIM-ONE Framework**
For more information: https://github.com/dansasser/SIM-ONE
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the FULL Five Laws protocols (not lightweight version)
import asyncio
try:
    from mcp_server.protocols.governance.five_laws_validator.law1_architectural_intelligence import ArchitecturalIntelligenceProtocol
    from mcp_server.protocols.governance.five_laws_validator.law2_cognitive_governance import CognitiveGovernanceProtocol
    from mcp_server.protocols.governance.five_laws_validator.law3_truth_foundation import TruthFoundationProtocol
    from mcp_server.protocols.governance.five_laws_validator.law4_energy_stewardship import EnergyStewardshipProtocol
    from mcp_server.protocols.governance.five_laws_validator.law5_deterministic_reliability import DeterministicReliabilityProtocol
    USE_FULL_PROTOCOLS = True
except ImportError as e:
    print(f"Warning: Could not import full protocols, falling back to lightweight evaluator: {e}", file=sys.stderr)
    try:
        from tools.lib.five_laws_evaluator import FiveLawsEvaluator, evaluate_text
        USE_FULL_PROTOCOLS = False
    except ImportError as e2:
        print(json.dumps({
            "error": f"Failed to import any Five Laws evaluator: {e2}",
            "status": "import_error"
        }), file=sys.stderr)
        sys.exit(1)


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def read_from_stdin() -> str:
    """Read input from stdin"""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def read_from_file(file_path: str) -> str:
    """Read input from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file: {e}")


def parse_context(context_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """Parse context JSON string"""
    if not context_str:
        return None
    try:
        return json.loads(context_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in context: {e}")


def _make_json_serializable(obj):
    """Convert non-serializable objects to JSON-compatible format"""
    if hasattr(obj, '__dict__'):
        # Convert objects with __dict__ to dicts
        return {k: _make_json_serializable(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, 'name'):
        # Convert Enums to their name
        return obj.name
    elif hasattr(obj, 'value'):
        # Convert Enums to their value
        return obj.value
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


def run_full_protocol_validation(text: str, context: Optional[Dict[str, Any]], strictness: str) -> Dict[str, Any]:
    """
    Run validation using the FULL Five Laws protocols (not lightweight version)

    This executes the actual 30-68KB protocol implementations with sophisticated logic.
    """
    # Prepare data for protocol execution
    protocol_data = {
        "cognitive_outputs": {"text": text, "content": text},
        "reasoning_chains": [],
        "knowledge_base": context if context else {},
        "validation_context": {"strictness": strictness}
    }

    # Initialize all Five Laws protocols
    law1 = ArchitecturalIntelligenceProtocol()
    law2 = CognitiveGovernanceProtocol()
    law3 = TruthFoundationProtocol()
    law4 = EnergyStewardshipProtocol()
    law5 = DeterministicReliabilityProtocol()

    # Run all protocols asynchronously
    async def execute_all_protocols():
        results = await asyncio.gather(
            law1.execute(protocol_data),
            law2.execute(protocol_data),
            law3.execute(protocol_data),
            law4.execute(protocol_data),
            law5.execute(protocol_data)
        )
        return results

    # Execute and gather results
    law_results = asyncio.run(execute_all_protocols())

    # Aggregate into expected output format (detailed_results excluded due to circular references)
    scores = {
        "law1_architectural_intelligence": law_results[0].get("compliance_score", 0) * 100,
        "law2_cognitive_governance": law_results[1].get("compliance_score", 0) * 100,
        "law3_truth_foundation": law_results[2].get("compliance_score", 0) * 100,
        "law4_energy_stewardship": law_results[3].get("compliance_score", 0) * 100,
        "law5_deterministic_reliability": law_results[4].get("compliance_score", 0) * 100,
    }

    # Weighted overall compliance
    scores["overall_compliance"] = (
        scores["law1_architectural_intelligence"] * 0.25 +
        scores["law2_cognitive_governance"] * 0.25 +
        scores["law3_truth_foundation"] * 0.25 +
        scores["law4_energy_stewardship"] * 0.15 +
        scores["law5_deterministic_reliability"] * 0.10
    )

    # Collect violations and recommendations from all protocols
    violations = []
    recommendations = []
    strengths = []

    for result in law_results:
        violations.extend(result.get("violations", []))
        recommendations.extend(result.get("recommendations", []))

    # Identify strengths
    law_names = ["Architectural Intelligence", "Cognitive Governance", "Truth Foundation",
                 "Energy Stewardship", "Deterministic Reliability"]
    law_scores = [scores[f"law{i+1}_{name.lower().replace(' ', '_')}"]
                  for i, name in enumerate(law_names)]

    for name, score in zip(law_names, law_scores):
        if score > 80:
            strengths.append(f"Strong {name} compliance")

    # Determine pass/fail status
    thresholds = {
        "lenient": {"pass": 60.0, "conditional": 45.0},
        "moderate": {"pass": 70.0, "conditional": 55.0},
        "strict": {"pass": 85.0, "conditional": 70.0}
    }
    threshold = thresholds.get(strictness, thresholds["moderate"])

    if scores["overall_compliance"] >= threshold["pass"]:
        pass_fail_status = "PASS"
    elif scores["overall_compliance"] >= threshold["conditional"]:
        pass_fail_status = "CONDITIONAL"
    else:
        pass_fail_status = "FAIL"

    return {
        "scores": scores,
        "violations": violations,
        "recommendations": recommendations,
        "strengths": strengths,
        "pass_fail_status": pass_fail_status
    }


def format_output(result: Dict[str, Any], format_type: str = "json") -> str:
    """Format output in specified format"""
    if format_type == "json":
        return json.dumps(result, indent=2)
    elif format_type == "compact":
        return json.dumps(result)
    elif format_type == "summary":
        # Human-readable summary
        scores = result["scores"]
        protocol_mode = result.get('protocol_mode', 'UNKNOWN')
        output = []
        output.append("=" * 80)
        output.append("Five Laws Cognitive Governance Validation Report")
        output.append("=" * 80)
        output.append(f"\nProtocol Mode: {protocol_mode}")
        output.append(f"Overall Compliance: {scores['overall_compliance']:.1f}%")
        output.append(f"Status: {result['pass_fail_status']}")
        output.append(f"\nIndividual Law Scores:")
        output.append(f"  1. Architectural Intelligence:    {scores['law1_architectural_intelligence']:.1f}%")
        output.append(f"  2. Cognitive Governance:          {scores['law2_cognitive_governance']:.1f}%")
        output.append(f"  3. Truth Foundation:              {scores['law3_truth_foundation']:.1f}%")
        output.append(f"  4. Energy Stewardship:            {scores['law4_energy_stewardship']:.1f}%")
        output.append(f"  5. Deterministic Reliability:     {scores['law5_deterministic_reliability']:.1f}%")

        if result.get("strengths"):
            output.append(f"\n[+] Strengths ({len(result['strengths'])}):")
            for strength in result["strengths"]:
                output.append(f"  - {strength}")

        if result.get("violations"):
            output.append(f"\n[-] Violations ({len(result['violations'])}):")
            for violation in result["violations"][:5]:  # Limit to first 5
                output.append(f"  - {violation}")
            if len(result["violations"]) > 5:
                output.append(f"  ... and {len(result['violations']) - 5} more")

        if result.get("recommendations"):
            output.append(f"\n[>] Recommendations ({len(result['recommendations'])}):")
            for rec in result["recommendations"][:5]:  # Limit to first 5
                output.append(f"  - {rec}")
            if len(result["recommendations"]) > 5:
                output.append(f"  ... and {len(result['recommendations']) - 5} more")

        output.append("\n" + "=" * 80)
        return "\n".join(output)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def main():
    """Main entry point for Five Laws validator tool"""
    parser = argparse.ArgumentParser(
        description="Validate AI responses against the Five Laws of Cognitive Governance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate text from command line
  python run_five_laws_validator.py --text "The Earth is round"

  # Validate from stdin
  echo "AI response" | python run_five_laws_validator.py

  # Validate from file
  python run_five_laws_validator.py --file response.txt

  # Use strict validation
  python run_five_laws_validator.py --text "..." --strictness strict

  # Get human-readable summary
  python run_five_laws_validator.py --text "..." --format summary

  # Include context
  python run_five_laws_validator.py --text "..." --context '{"domain": "scientific"}'
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Text to validate (as command line argument)'
    )
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='File containing text to validate'
    )

    # Configuration options
    parser.add_argument(
        '--strictness', '-s',
        type=str,
        choices=['lenient', 'moderate', 'strict'],
        default='moderate',
        help='Validation strictness level (default: moderate)'
    )
    parser.add_argument(
        '--context', '-c',
        type=str,
        help='Additional context as JSON string'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'compact', 'summary'],
        default='json',
        help='Output format (default: json)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Get input text
        if args.text:
            text = args.text
            logger.debug("Using text from command line argument")
        elif args.file:
            text = read_from_file(args.file)
            logger.debug(f"Read {len(text)} chars from file: {args.file}")
        else:
            # Try stdin
            text = read_from_stdin()
            if not text:
                parser.print_help()
                print("\nError: No input provided. Use --text, --file, or pipe via stdin.",
                      file=sys.stderr)
                sys.exit(1)
            logger.debug(f"Read {len(text)} chars from stdin")

        # Validate input
        if not text or len(text.strip()) == 0:
            print(json.dumps({
                "error": "Empty input provided",
                "status": "invalid_input"
            }), file=sys.stderr)
            sys.exit(1)

        # Parse context
        context = parse_context(args.context)
        if context:
            logger.debug(f"Using context: {context}")

        # Run evaluation using full protocols if available, otherwise use lightweight
        logger.info(f"Evaluating {len(text)} chars with strictness={args.strictness}")
        if USE_FULL_PROTOCOLS:
            logger.info("Using FULL Five Laws protocols (30-68KB implementations)")
            result = run_full_protocol_validation(text, context, args.strictness)
            result['protocol_mode'] = 'FULL_PROTOCOLS'
        else:
            logger.info("Using lightweight evaluator (fallback)")
            result = evaluate_text(text, context, strictness=args.strictness)
            result['protocol_mode'] = 'LIGHTWEIGHT_FALLBACK'

        # Format and output result
        output = format_output(result, args.format)
        print(output)

        # Exit code based on status
        status = result.get("pass_fail_status", "UNKNOWN")
        if status == "PASS":
            sys.exit(0)
        elif status == "CONDITIONAL":
            sys.exit(0)  # Still exit 0, but caller can check status in JSON
        elif status == "FAIL":
            sys.exit(1)
        else:
            sys.exit(2)  # Error or unknown status

    except ValueError as e:
        print(json.dumps({
            "error": str(e),
            "status": "validation_error"
        }), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error during evaluation")
        print(json.dumps({
            "error": f"Unexpected error: {e}",
            "status": "internal_error"
        }), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
