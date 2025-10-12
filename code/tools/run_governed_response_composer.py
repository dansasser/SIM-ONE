#!/usr/bin/env python3
"""
Tool: Governed Response Composer
Input: User prompt, governance threshold, max iterations, strictness level
Output: AI-generated response that meets SIM-ONE governance standards

MCP_TOOL_ENTRYPOINT

This tool generates AI responses using Claude Code CLI (claude -p) and iteratively refines
them using SIM-ONE's Five Laws of Cognitive Governance validation until the governance
threshold is met or maximum iterations are reached.

## How It Works

1. **Initial Generation**: Uses `claude -p` to generate an initial response, instructing
   Claude to consult the SIM-ONE Framework documentation via MCP connector
   
2. **Validation**: Validates the response against the Five Laws of Cognitive Governance

3. **Iterative Refinement**: If validation score is below threshold, creates a targeted
   refinement prompt identifying specific governance issues and generates an improved response

4. **Convergence**: Repeats refinement until threshold is met or max iterations reached

## Five Laws of Cognitive Governance

1. **Architectural Intelligence** - Intelligence from coordination, not brute force
2. **Cognitive Governance** - Governed processes over unconstrained generation
3. **Truth Foundation** - Absolute truth principles over probabilistic drift
4. **Energy Stewardship** - Computational efficiency and resource awareness
5. **Deterministic Reliability** - Consistent, predictable outcomes

## Usage

### Basic usage:
```bash
python run_governed_response_composer.py --prompt "Explain quantum computing"
```

### With custom threshold:
```bash
python run_governed_response_composer.py \
    --prompt "Explain quantum computing" \
    --threshold 85.0
```

### With strictness level:
```bash
python run_governed_response_composer.py \
    --prompt "Explain quantum computing" \
    --strictness strict
```

### Save to file:
```bash
python run_governed_response_composer.py \
    --prompt "Explain quantum computing" \
    --output result.json
```

### Full options:
```bash
python run_governed_response_composer.py \
    --prompt "Your prompt here" \
    --threshold 80.0 \
    --max-iterations 3 \
    --strictness moderate \
    --output result.json \
    --verbose
```

## Output Format

JSON structure:
```json
{
  "initial_response": {
    "text": "Initial AI response",
    "validation": {
      "overall_score": 75.5,
      "law_scores": {...},
      "violations": [...],
      "recommendations": [...],
      "passed": false
    }
  },
  "final_response": {
    "text": "Refined AI response",
    "validation": {
      "overall_score": 82.3,
      "law_scores": {...},
      "violations": [...],
      "recommendations": [...],
      "passed": true
    }
  },
  "iterations": 2,
  "passed": true,
  "improvement": 6.8
}
```

## Requirements

- Claude Code CLI (`claude`) must be installed and authenticated
- SIM-ONE Framework protocols must be available
- Python 3.8+

---

**Part of the SIM-ONE Framework**
For more information: https://github.com/dansasser/SIM-ONE
"""

import sys
import json
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Literal, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Five Laws validation from existing wrapper
from tools.run_five_laws_validator import run_full_protocol_validation

# Configuration
DEFAULT_THRESHOLD = 80.0
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_STRICTNESS = "moderate"
CLAUDE_TIMEOUT = 120  # 2 minutes


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def _clean_unicode(text_list: List[str]) -> List[str]:
    """
    Remove Unicode characters for Windows compatibility.
    
    Replaces common Unicode symbols with ASCII equivalents:
    - ✓ → [+]
    - ✗ → [-]
    - • → *
    - → → ->
    
    Args:
        text_list: List of strings that may contain Unicode
        
    Returns:
        List of strings with Unicode replaced
    """
    cleaned = []
    for text in text_list:
        text = text.replace('✓', '[+]')
        text = text.replace('✗', '[-]')
        text = text.replace('•', '*')
        text = text.replace('→', '->')
        text = text.replace('—', '--')
        text = text.replace(''', "'")
        text = text.replace(''', "'")
        text = text.replace('"', '"')
        text = text.replace('"', '"')
        cleaned.append(text)
    return cleaned


def _generate_initial_response(prompt: str) -> str:
    """
    Generate initial response using claude -p with MCP-aware prompt.
    
    Instructs Claude to consult SIM-ONE Framework documentation via MCP connector
    before generating the response.
    
    Args:
        prompt: User's original prompt
        
    Returns:
        Generated response text
        
    Raises:
        FileNotFoundError: If claude CLI is not installed
        subprocess.TimeoutExpired: If response generation times out
        RuntimeError: If claude CLI returns an error
    """
    # Build MCP-aware prompt
    full_prompt = f"""Before responding, use the SIM-ONE MCP connector to review the latest SIM-ONE Framework documentation, especially the Five Laws of Cognitive Governance.

User request: {prompt}

Generate a response that adheres to the SIM-ONE Framework's Five Laws of Cognitive Governance."""

    logging.debug(f"Generating initial response with prompt: {full_prompt[:100]}...")
    
    try:
        # Execute claude -p
        result = subprocess.run(
            ["claude", "-p", full_prompt],
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI error: {error_msg}")
        
        # Parse JSON output
        try:
            response_data = json.loads(result.stdout)
            # Extract text from Claude API format
            if "content" in response_data and len(response_data["content"]) > 0:
                return response_data["content"][0]["text"]
            else:
                raise ValueError("Unexpected response format from Claude CLI")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude response: {result.stdout[:200]}")
            raise RuntimeError(f"Invalid JSON response from Claude CLI: {e}")
            
    except FileNotFoundError:
        raise FileNotFoundError(
            "Claude CLI not found. Please install it with: pip install claude-cli"
        )
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(
            cmd=["claude", "-p", "..."],
            timeout=CLAUDE_TIMEOUT
        )


def _generate_refined_response(refinement_prompt: str) -> str:
    """
    Generate refined response using claude -p with refinement prompt.
    
    Args:
        refinement_prompt: Prompt with validation feedback and refinement instructions
        
    Returns:
        Refined response text
        
    Raises:
        Same as _generate_initial_response
    """
    logging.debug(f"Generating refined response with prompt: {refinement_prompt[:100]}...")
    
    try:
        # Execute claude -p
        result = subprocess.run(
            ["claude", "-p", refinement_prompt],
            capture_output=True,
            text=True,
            timeout=CLAUDE_TIMEOUT
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            raise RuntimeError(f"Claude CLI error: {error_msg}")
        
        # Parse JSON output
        try:
            response_data = json.loads(result.stdout)
            if "content" in response_data and len(response_data["content"]) > 0:
                return response_data["content"][0]["text"]
            else:
                raise ValueError("Unexpected response format from Claude CLI")
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude response: {result.stdout[:200]}")
            raise RuntimeError(f"Invalid JSON response from Claude CLI: {e}")
            
    except FileNotFoundError:
        raise FileNotFoundError(
            "Claude CLI not found. Please install it with: pip install claude-cli"
        )
    except subprocess.TimeoutExpired:
        raise subprocess.TimeoutExpired(
            cmd=["claude", "-p", "..."],
            timeout=CLAUDE_TIMEOUT
        )


def _validate_response(text: str, strictness: str) -> Dict[str, Any]:
    """
    Validate response using Five Laws validation.
    
    Args:
        text: Response text to validate
        strictness: Validation strictness level
        
    Returns:
        Validation result dict with scores, violations, recommendations
    """
    logging.debug(f"Validating response (strictness: {strictness})")
    
    # Call existing Five Laws validation
    result = run_full_protocol_validation(text, context=None, strictness=strictness)
    
    # Clean Unicode from violations and recommendations
    if "violations" in result:
        result["violations"] = _clean_unicode(result["violations"])
    if "recommendations" in result:
        result["recommendations"] = _clean_unicode(result["recommendations"])
    
    # Extract key metrics
    validation_result = {
        "overall_score": result["scores"]["overall_compliance"],
        "law_scores": result["scores"],
        "violations": result["violations"],
        "recommendations": result["recommendations"],
        "passed": result["pass_fail_status"] == "PASS"
    }
    
    logging.debug(f"Validation score: {validation_result['overall_score']:.1f}%")
    
    return validation_result


def _create_refinement_prompt(
    original_prompt: str,
    previous_response: str,
    validation_result: Dict[str, Any]
) -> str:
    """
    Create targeted refinement prompt based on validation results.
    
    Identifies which laws failed and creates a prompt that instructs Claude
    to consult specific sections of the SIM-ONE Framework documentation.
    
    Args:
        original_prompt: User's original prompt
        previous_response: Previous response that failed validation
        validation_result: Validation result with scores and violations
        
    Returns:
        Refinement prompt string
    """
    # Identify failed laws (score < 80)
    law_names = {
        "law1_architectural_intelligence": "Law 1 (Architectural Intelligence)",
        "law2_cognitive_governance": "Law 2 (Cognitive Governance)",
        "law3_truth_foundation": "Law 3 (Truth Foundation)",
        "law4_energy_stewardship": "Law 4 (Energy Stewardship)",
        "law5_deterministic_reliability": "Law 5 (Deterministic Reliability)"
    }
    
    failed_laws = []
    for law_key, law_name in law_names.items():
        if law_key in validation_result["law_scores"]:
            score = validation_result["law_scores"][law_key]
            if score < 80.0:
                failed_laws.append(f"{law_name} (score: {score:.1f}%)")
    
    # Build targeted refinement prompt
    laws_to_review = "\n".join([f"- {law}" for law in failed_laws]) if failed_laws else "- All Five Laws"
    
    violations_text = "\n".join([f"- {v}" for v in validation_result["violations"][:5]])  # Limit to top 5
    recommendations_text = "\n".join([f"- {r}" for r in validation_result["recommendations"][:5]])
    
    refinement_prompt = f"""Before responding, use the SIM-ONE MCP connector to review the latest SIM-ONE Framework documentation, especially regarding:
{laws_to_review}

Original user request: {original_prompt}

Previous response:
{previous_response}

Governance validation score: {validation_result['overall_score']:.1f}%

Issues found:
{violations_text}

Recommendations:
{recommendations_text}

Please revise the response using the SIM-ONE Framework documentation to address these specific governance issues and improve the score above 80%."""

    return refinement_prompt


def compose_governed_response(
    prompt: str,
    threshold: float = DEFAULT_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    strictness: Literal["lenient", "moderate", "strict"] = DEFAULT_STRICTNESS
) -> Dict[str, Any]:
    """
    Generate AI response that meets SIM-ONE governance standards.
    
    This function uses claude -p to generate responses and iteratively refines them
    using Five Laws validation until the governance threshold is met or maximum
    iterations are reached.
    
    Args:
        prompt: User prompt for response generation
        threshold: Minimum governance score (0-100) required to pass
        max_iterations: Maximum number of refinement iterations
        strictness: Validation strictness level (lenient/moderate/strict)
        
    Returns:
        Dict containing:
        - initial_response: {text, validation} of first generation
        - final_response: {text, validation} of final generation
        - iterations: Number of refinement iterations performed
        - passed: Whether final response met threshold
        - improvement: Score improvement from initial to final
        
    Raises:
        FileNotFoundError: If claude CLI is not installed
        subprocess.TimeoutExpired: If response generation times out
        RuntimeError: If claude CLI returns an error
        
    Example:
        >>> result = compose_governed_response(
        ...     prompt="Explain quantum computing",
        ...     threshold=80.0,
        ...     max_iterations=3,
        ...     strictness="moderate"
        ... )
        >>> print(f"Score: {result['final_response']['validation']['overall_score']:.1f}%")
        >>> print(f"Iterations: {result['iterations']}")
    """
    logging.info(f"Starting governed response composition for prompt: {prompt[:50]}...")
    logging.info(f"Threshold: {threshold}%, Max iterations: {max_iterations}, Strictness: {strictness}")
    
    # 1. Generate initial response
    logging.info("Generating initial response...")
    initial_response_text = _generate_initial_response(prompt)
    initial_validation = _validate_response(initial_response_text, strictness)
    
    initial_response = {
        "text": initial_response_text,
        "validation": initial_validation
    }
    
    logging.info(f"Initial response score: {initial_validation['overall_score']:.1f}%")
    
    # 2. Check if already passed
    if initial_validation["overall_score"] >= threshold:
        logging.info("Initial response already meets threshold!")
        return {
            "initial_response": initial_response,
            "final_response": initial_response,
            "iterations": 0,
            "passed": True,
            "improvement": 0.0
        }
    
    # 3. Iterative refinement
    current_response_text = initial_response_text
    current_validation = initial_validation
    iteration = 0
    
    while iteration < max_iterations and current_validation["overall_score"] < threshold:
        iteration += 1
        logging.info(f"Starting refinement iteration {iteration}/{max_iterations}...")
        
        # Create refinement prompt
        refinement_prompt = _create_refinement_prompt(
            prompt, current_response_text, current_validation
        )
        
        # Generate refined response
        current_response_text = _generate_refined_response(refinement_prompt)
        
        # Validate refined response
        current_validation = _validate_response(current_response_text, strictness)
        
        logging.info(f"Iteration {iteration} score: {current_validation['overall_score']:.1f}%")
        
        # Check if passed
        if current_validation["overall_score"] >= threshold:
            logging.info(f"Threshold met after {iteration} iteration(s)!")
            break
    
    # 4. Return comparison
    final_response = {
        "text": current_response_text,
        "validation": current_validation
    }
    
    improvement = current_validation["overall_score"] - initial_validation["overall_score"]
    passed = current_validation["overall_score"] >= threshold
    
    logging.info(f"Final score: {current_validation['overall_score']:.1f}% (improvement: {improvement:+.1f}%)")
    logging.info(f"Status: {'PASSED' if passed else 'FAILED'}")
    
    return {
        "initial_response": initial_response,
        "final_response": final_response,
        "iterations": iteration,
        "passed": passed,
        "improvement": improvement
    }


def main():
    """CLI interface for governed response composer"""
    parser = argparse.ArgumentParser(
        description="Generate AI responses that meet SIM-ONE governance standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_governed_response_composer.py --prompt "Explain quantum computing"
  
  # With custom threshold
  python run_governed_response_composer.py --prompt "..." --threshold 85.0
  
  # Save to file
  python run_governed_response_composer.py --prompt "..." --output result.json
  
  # Verbose output
  python run_governed_response_composer.py --prompt "..." --verbose
        """
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt for response generation"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum governance score (0-100) required to pass (default: {DEFAULT_THRESHOLD})"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum number of refinement iterations (default: {DEFAULT_MAX_ITERATIONS})"
    )
    
    parser.add_argument(
        "--strictness",
        type=str,
        choices=["lenient", "moderate", "strict"],
        default=DEFAULT_STRICTNESS,
        help=f"Validation strictness level (default: {DEFAULT_STRICTNESS})"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSON file path to save results"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Execute governed response composition
        result = compose_governed_response(
            prompt=args.prompt,
            threshold=args.threshold,
            max_iterations=args.max_iterations,
            strictness=args.strictness
        )
        
        # Output handling
        output_json = json.dumps(result, indent=2)
        
        if args.output:
            # Save to file
            output_path = Path(args.output)
            output_path.write_text(output_json)
            print(f"Results saved to: {args.output}", file=sys.stderr)
        else:
            # Print to stdout
            print(output_json)
        
        # Exit with appropriate code
        sys.exit(0 if result["passed"] else 1)
        
    except FileNotFoundError as e:
        print(json.dumps({
            "error": str(e),
            "status": "claude_cli_not_found",
            "help": "Install Claude CLI: pip install claude-cli"
        }), file=sys.stderr)
        sys.exit(2)
        
    except subprocess.TimeoutExpired:
        print(json.dumps({
            "error": f"Response generation timed out after {CLAUDE_TIMEOUT} seconds",
            "status": "timeout"
        }), file=sys.stderr)
        sys.exit(3)
        
    except Exception as e:
        logging.exception("Unexpected error")
        print(json.dumps({
            "error": str(e),
            "status": "error"
        }), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

