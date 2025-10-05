# SIM-ONE Tools Catalog

**Cognitive Governance Tools for Autonomous AI Systems**

This directory contains CLI tool wrappers for SIM-ONE protocols, enabling Paper2Agent and other autonomous systems to discover and use governed cognitive processes.

---

## Quick Reference

| Tool | Category | Purpose | Priority |
|------|----------|---------|----------|
| `run_five_laws_validator.py` | Governance | Validate text against Five Laws | ⭐ Highest |
| `run_rep_tool.py` | Reasoning | Multi-modal reasoning & explanation | High |
| `run_esl_tool.py` | Emotional Intelligence | Emotion detection & analysis | Medium |
| `run_vvp_tool.py` | Validation | Input validation & verification | High |

---

## Table of Contents

- [Overview](#overview)
- [Tool Discovery](#tool-discovery)
- [Governance Tools](#governance-tools)
- [Protocol Tools](#protocol-tools)
- [Usage Patterns](#usage-patterns)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

These tools enable autonomous AI systems to:
- **Self-govern** their outputs using the Five Laws of Cognitive Governance
- **Compose protocols** to create complex governed workflows
- **Validate** inputs and outputs for quality assurance
- **Reason systematically** using multiple reasoning modes
- **Analyze emotions** for empathy-aware interactions

### Tool Conventions

All tools follow consistent patterns:

- **Naming**: `run_*_tool.py`
- **Marker**: Contains `# MCP_TOOL_ENTRYPOINT` for discovery
- **Input**: Support CLI args, files, or stdin
- **Output**: JSON by default (some support summary format)
- **Exit Codes**: `0` = success, `1` = failure, `2` = error
- **Composable**: Can be chained via pipes

---

## Tool Discovery

### Manifest File

All tools are cataloged in `tools_manifest.json`:

```bash
# View all available tools
cat tools_manifest.json | jq '.governance_tools, .protocol_tools'

# Get tool count
cat tools_manifest.json | jq '.metadata.total_tools'
```

### Discovery Pattern

```python
import json

# Load manifest
with open('tools_manifest.json') as f:
    manifest = json.load(f)

# List all governance tools
for tool_name, tool_info in manifest['governance_tools'].items():
    print(f"{tool_name}: {tool_info['description']}")

# Get tool wrapper path
wrapper = manifest['governance_tools']['five_laws_validator']['wrapper']
print(f"Tool path: {wrapper}")
```

---

## Governance Tools

### Five Laws Validator ⭐

**File**: `run_five_laws_validator.py`

**Purpose**: Validate any AI-generated text against the Five Laws of Cognitive Governance

**Quick Start:**
```bash
# Validate text
python run_five_laws_validator.py --text "Response to validate"

# From file
python run_five_laws_validator.py --file response.txt

# From stdin
echo "Response" | python run_five_laws_validator.py

# Human-readable summary
python run_five_laws_validator.py --text "..." --format summary

# Strict validation
python run_five_laws_validator.py --text "..." --strictness strict
```

**Options:**
| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--text, -t` | string | Text to validate | - |
| `--file, -f` | path | File containing text | - |
| `--strictness, -s` | choice | lenient\|moderate\|strict | moderate |
| `--context, -c` | json | Additional context | - |
| `--format` | choice | json\|compact\|summary | json |
| `--verbose, -v` | flag | Verbose logging | false |

**Output Structure:**
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
  "violations": ["List of violations"],
  "recommendations": ["Actionable improvements"],
  "strengths": ["Identified strengths"],
  "detailed_results": {...}
}
```

**Exit Codes:**
- `0`: PASS or CONDITIONAL
- `1`: FAIL
- `2`: ERROR

**Use Cases:**
- Pre-response validation for AI systems
- Quality assurance in content pipelines
- Governance auditing
- Multi-agent validation

---

## Protocol Tools

### REP - Reasoning & Explanation Protocol

**File**: `run_rep_tool.py`

**Purpose**: Advanced multi-modal reasoning including deductive, inductive, abductive, analogical, and causal inference

**Quick Start:**
```bash
# Deductive reasoning
python run_rep_tool.py --reasoning-type deductive \
  --facts "Socrates is a man" "All men are mortal" \
  --rules '[["Socrates is a man", "All men are mortal"], "Socrates is mortal"]'

# From JSON file
python run_rep_tool.py --file reasoning_input.json

# From stdin
echo '{"reasoning_type": "inductive", "observations": [...]}' | python run_rep_tool.py
```

**Reasoning Types:**
- **deductive**: Logical deduction (modus ponens, modus tollens)
- **inductive**: Pattern recognition and generalization
- **abductive**: Best explanation inference
- **analogical**: Similarity-based reasoning
- **causal**: Cause-effect relationship analysis

**Input Format (JSON):**
```json
{
  "reasoning_type": "deductive",
  "facts": ["fact1", "fact2"],
  "rules": [[["premise1", "premise2"], "conclusion"]],
  "context": "optional context"
}
```

**Output:**
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

**Use Cases:**
- Logical inference and deduction
- Pattern recognition from observations
- Hypothesis selection
- Similarity-based reasoning
- Causal analysis

---

### ESL - Emotional State Layer

**File**: `run_esl_tool.py`

**Purpose**: Sophisticated multi-dimensional emotion detection and analysis

**Quick Start:**
```bash
# Analyze emotional content
python run_esl_tool.py --text "I'm absolutely thrilled about this!"

# From file
python run_esl_tool.py --file user_message.txt

# Summary format
python run_esl_tool.py --text "..." --format summary
```

**Emotion Dimensions:**
- **Primary**: joy, sadness, anger, fear, surprise, disgust
- **Social**: gratitude, empathy, pride, guilt, envy
- **Cognitive**: hope, curiosity, confusion, confidence, doubt

**Output:**
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

**Use Cases:**
- Sentiment analysis
- Empathy-aware communication
- User state detection
- Adaptive response generation

---

### VVP - Validation & Verification Protocol

**File**: `run_vvp_tool.py`

**Purpose**: Input validation and verification for logical structures

**Quick Start:**
```bash
# Validate rule structures
python run_vvp_tool.py --json '{"rules": [[["a"], "b"], [["b"], "c"]]}'

# From file
python run_vvp_tool.py --file validation_input.json
```

**Input Format:**
```json
{
  "rules": [
    [["premise1", "premise2"], "conclusion"]
  ]
}
```

**Output:**
```json
{
  "validation_status": "success",
  "reason": "Input validation passed: 2 rules checked and well-formed"
}
```

**Use Cases:**
- Pre-processing validation
- Rule structure verification
- Logical consistency checking
- Input format compliance

---

## Usage Patterns

### Pattern 1: Simple Validation

```bash
# Validate a single response
python run_five_laws_validator.py --text "The Earth is round"
```

### Pattern 2: File-Based Processing

```bash
# Process file
python run_five_laws_validator.py --file ai_response.txt > validation_result.json
```

### Pattern 3: Pipeline Composition

```bash
# Chain multiple tools
python run_rep_tool.py --json '{"reasoning_type": "deductive", ...}' | \
  python run_vvp_tool.py | \
  python run_five_laws_validator.py
```

### Pattern 4: Conditional Execution

```bash
#!/bin/bash
# Validate before proceeding

validation=$(echo "$response" | python run_five_laws_validator.py)
status=$(echo "$validation" | jq -r '.pass_fail_status')

if [ "$status" == "PASS" ]; then
    echo "✅ Response is governed"
    proceed_with_response "$response"
else
    echo "❌ Response needs refinement"
    refine_response "$response" "$(echo "$validation" | jq '.recommendations')"
fi
```

### Pattern 5: Iterative Refinement

```python
def refine_until_governed(initial_response, max_iterations=3):
    """Keep refining response until Five Laws compliance"""
    import subprocess
    import json

    response = initial_response

    for i in range(max_iterations):
        # Validate
        result = subprocess.run(
            ["python", "run_five_laws_validator.py", "--text", response],
            capture_output=True, text=True
        )
        validation = json.loads(result.stdout)

        # Check status
        if validation["pass_fail_status"] == "PASS":
            print(f"✅ Achieved governance in {i+1} iterations")
            return response

        # Refine based on recommendations
        response = apply_recommendations(response, validation["recommendations"])

    return response
```

---

## Examples

### Example 1: Validate ChatGPT Response

```bash
# Get response from ChatGPT
response="Climate change is real based on scientific consensus"

# Validate
echo "$response" | python run_five_laws_validator.py --format summary
```

### Example 2: Governed Response Generation

```python
#!/usr/bin/env python3
import subprocess
import json

def generate_governed_response(prompt):
    """Generate response that passes Five Laws"""

    # Your LLM generation
    response = your_llm_generate(prompt)

    # Validate
    result = subprocess.run(
        ["python", "code/tools/run_five_laws_validator.py", "--text", response],
        capture_output=True, text=True
    )
    validation = json.loads(result.stdout)

    # Return response + governance metadata
    return {
        "response": response,
        "governance_score": validation["scores"]["overall_compliance"],
        "status": validation["pass_fail_status"]
    }

# Usage
result = generate_governed_response("Explain quantum mechanics")
print(f"Response (Score: {result['governance_score']:.1f}%): {result['response']}")
```

### Example 3: Multi-Step Workflow

```bash
#!/bin/bash
# Complete governed workflow

# Step 1: Analyze user emotion
emotion=$(python run_esl_tool.py --text "$user_input")

# Step 2: Generate empathetic response
response=$(generate_response_with_emotion "$user_input" "$emotion")

# Step 3: Apply reasoning
reasoning=$(python run_rep_tool.py --json "{\"reasoning_type\": \"deductive\", ...}")

# Step 4: Validate
python run_vvp_tool.py --json "$reasoning"

# Step 5: Five Laws governance check
python run_five_laws_validator.py --text "$response"
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'mcp_server'`

**Solution**:
```bash
# Ensure you're in code/tools directory
cd code/tools

# Or run with full path
python /full/path/to/code/tools/run_five_laws_validator.py

# Install dependencies
cd ../
pip install -r requirements.txt
```

#### 2. Tool Not Found

**Problem**: `command not found: run_five_laws_validator.py`

**Solution**:
```bash
# Use python explicitly
python run_five_laws_validator.py --text "..."

# Or make executable and add to PATH
chmod +x run_five_laws_validator.py
export PATH=$PATH:/path/to/code/tools
```

#### 3. Low Compliance Scores

**Problem**: Consistently getting FAIL status

**Solution**:
```bash
# Use summary format to see specific violations
python run_five_laws_validator.py --text "..." --format summary

# Review recommendations
python run_five_laws_validator.py --text "..." | jq '.recommendations[]'

# Try lenient strictness first
python run_five_laws_validator.py --text "..." --strictness lenient
```

#### 4. JSON Parsing Errors

**Problem**: `json.decoder.JSONDecodeError`

**Solution**:
```bash
# Validate JSON before piping
echo '{"rules": [[["a"], "b"]]}' | jq '.' | python run_vvp_tool.py

# Use --json flag for complex structures
python run_rep_tool.py --json '{"reasoning_type": "deductive", ...}'
```

---

## Development

### Adding New Tools

To add a new protocol wrapper:

1. **Create wrapper file**: `run_[protocol]_tool.py`
2. **Add entrypoint marker**: `# MCP_TOOL_ENTRYPOINT`
3. **Follow I/O conventions**: Support `--file`, `--json`, stdin
4. **Update manifest**: Add entry to `tools_manifest.json`
5. **Document usage**: Add section to this README
6. **Test**: Ensure all input methods work

**Template**:
```python
#!/usr/bin/env python3
"""
Tool: [Protocol Name]
Input: [Input description]
Output: [Output description]

MCP_TOOL_ENTRYPOINT

[Detailed description]
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_server.protocols.[name].[name] import ProtocolClass

def main():
    parser = argparse.ArgumentParser(description="[Protocol Name]")

    # Add arguments
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--file', '-f', help='Input file')
    input_group.add_argument('--json', '-j', help='JSON input')

    args = parser.parse_args()

    # Get input
    if args.file:
        with open(args.file) as f:
            data = json.load(f)
    elif args.json:
        data = json.loads(args.json)
    elif not sys.stdin.isatty():
        data = json.load(sys.stdin)
    else:
        parser.print_help()
        sys.exit(1)

    # Execute protocol
    protocol = ProtocolClass()
    result = protocol.execute(data)

    # Output
    print(json.dumps(result, indent=2))
    sys.exit(0)

if __name__ == "__main__":
    main()
```

---

## Testing

### Manual Testing

```bash
# Test Five Laws Validator
python run_five_laws_validator.py --text "Test response" --format summary

# Test REP
python run_rep_tool.py --reasoning-type deductive \
  --facts "a" "b" --rules '[["a", "b"], "c"]'

# Test ESL
python run_esl_tool.py --text "I'm happy!" --format summary

# Test VVP
python run_vvp_tool.py --json '{"rules": [[["a"], "b"]]}'
```

### Automated Testing

```bash
# Run test suite (when available)
cd ../tests
pytest test_tool_wrappers.py -v
```

---

## Performance

### Tool Execution Times

| Tool | Avg Execution | Complexity |
|------|---------------|------------|
| Five Laws Validator | ~200ms | Medium |
| REP | ~150ms | Low-Medium |
| ESL | ~100ms | Low |
| VVP | ~50ms | Very Low |

*Times measured on standard hardware with moderate input size*

### Optimization Tips

1. **Use compact output** for faster parsing:
   ```bash
   python run_five_laws_validator.py --format compact
   ```

2. **Batch processing** via files instead of multiple calls:
   ```bash
   # Slower
   for text in $texts; do
       python run_five_laws_validator.py --text "$text"
   done

   # Faster
   cat all_texts.txt | python run_five_laws_validator.py
   ```

3. **Cache results** for repeated validations of identical text

---

## Resources

- **Integration Guide**: [../PAPER2AGENT_INTEGRATION.md](../../PAPER2AGENT_INTEGRATION.md)
- **Tools Manifest**: [tools_manifest.json](tools_manifest.json)
- **Implementation Plan**: [../../docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md](../../docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md)
- **Five Laws Documentation**: [../../docs/five_laws/](../../docs/five_laws/)
- **GitHub Repository**: https://github.com/dansasser/SIM-ONE

---

## Support

For issues, questions, or contributions:
- **GitHub Issues**: https://github.com/dansasser/SIM-ONE/issues
- **Documentation**: See `PAPER2AGENT_INTEGRATION.md`
- **Examples**: See `examples/` directory

---

**Last Updated**: 2025-01-10 (Phase 22)

*Part of the SIM-ONE Framework - Cognitive Governance for Autonomous AI*
