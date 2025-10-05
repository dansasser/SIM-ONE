# Paper2Agent Integration Guide

**SIM-ONE Framework for Autonomous AI Governance**

This guide explains how Paper2Agent and other autonomous AI systems can discover and use SIM-ONE's cognitive governance protocols to self-govern their outputs and ensure compliance with the Five Laws of Cognitive Governance.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Five Laws Governance](#five-laws-governance)
- [Tool Discovery](#tool-discovery)
- [Available Tools](#available-tools)
- [Example Scenarios](#example-scenarios)
- [Protocol Composition](#protocol-composition)
- [API Reference](#api-reference)
- [Integration Patterns](#integration-patterns)

---

## Quick Start

### 1. Validate Any AI Response (One Command)

```bash
# Validate a response against the Five Laws
echo "The Earth is round based on scientific evidence" | \
  python code/tools/run_five_laws_validator.py --format summary
```

**Output:**
```
================================================================================
Five Laws Cognitive Governance Validation Report
================================================================================

Overall Compliance: 82.5%
Status: PASS

Individual Law Scores:
  1. Architectural Intelligence:    75.0%
  2. Cognitive Governance:          80.0%
  3. Truth Foundation:              95.0%
  4. Energy Stewardship:            72.0%
  5. Deterministic Reliability:     85.0%

✓ Strengths (2):
  • Solid grounding in truth principles
  • Consistent and reliable outputs

→ Recommendations (1):
  • Use multiple specialized protocols in coordination
================================================================================
```

### 2. Programmatic Integration

```python
import subprocess
import json

def validate_response(ai_response):
    """Validate AI response against Five Laws"""
    result = subprocess.run(
        ["python", "code/tools/run_five_laws_validator.py", "--text", ai_response],
        capture_output=True,
        text=True
    )

    return json.loads(result.stdout)

# Example usage
response = "Climate change is real based on overwhelming scientific consensus"
validation = validate_response(response)

if validation["pass_fail_status"] == "PASS":
    print(f"✅ Response is governed (Score: {validation['scores']['overall_compliance']:.1f}%)")
else:
    print(f"❌ Response needs refinement")
    for rec in validation["recommendations"]:
        print(f"  • {rec}")
```

---

## Five Laws Governance

### The Five Laws of Cognitive Governance

SIM-ONE's governance framework is built on five fundamental principles:

1. **Architectural Intelligence** - Intelligence emerges from coordination and governance, not from model size or brute force
2. **Cognitive Governance** - Every cognitive process must be governed by specialized protocols
3. **Truth Foundation** - All reasoning must be grounded in absolute truth principles, not probabilistic drift
4. **Energy Stewardship** - Computational resources are finite and must be stewarded responsibly
5. **Deterministic Reliability** - Governed systems must produce consistent, predictable, verifiable outcomes

### Why This Matters for Paper2Agent

Traditional AI systems generate outputs without governance constraints, leading to:
- ❌ Probabilistic drift and hallucinations
- ❌ Inconsistent quality across responses
- ❌ Lack of verifiable reasoning chains
- ❌ Uncontrolled resource consumption
- ❌ Non-reproducible results

**SIM-ONE's Five Laws governance enables Paper2Agent to:**
- ✅ Self-validate outputs before returning to users
- ✅ Detect and prevent truth foundation violations
- ✅ Ensure consistent, governed reasoning
- ✅ Optimize resource efficiency
- ✅ Provide verifiable compliance metrics

---

## Tool Discovery

### Finding Available Tools

Paper2Agent can discover all SIM-ONE tools via the tools manifest:

```bash
# Read the tools manifest
cat code/tools/tools_manifest.json
```

**Manifest Structure:**
```json
{
  "governance_tools": {
    "five_laws_validator": {
      "name": "Five Laws Cognitive Governance Validator",
      "wrapper": "run_five_laws_validator.py",
      "category": "governance",
      "priority": "highest",
      "input": {...},
      "output": {...},
      "use_cases": [...]
    }
  },
  "protocol_tools": {
    "REP": {...},
    "ESL": {...},
    "VVP": {...}
  }
}
```

### Tool Naming Convention

All SIM-ONE tool wrappers follow a consistent pattern:

- **Location**: `code/tools/`
- **Naming**: `run_*_tool.py`
- **Marker**: Contains `# MCP_TOOL_ENTRYPOINT` comment
- **Input**: Support CLI args, files, or stdin
- **Output**: JSON by default

---

## Available Tools

### 1. Five Laws Validator ⭐ (Highest Priority)

**Purpose**: Validate any AI-generated text against all Five Laws

**Usage:**
```bash
# From command line
python run_five_laws_validator.py --text "Response to validate"

# From file
python run_five_laws_validator.py --file response.txt

# From stdin (for piping)
echo "Response" | python run_five_laws_validator.py

# With strictness control
python run_five_laws_validator.py --text "..." --strictness strict
```

**Input:**
- Text to validate (required)
- Strictness level: `lenient` | `moderate` | `strict` (optional)
- Context JSON (optional)

**Output (JSON):**
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
  "violations": ["Specific violations listed here"],
  "recommendations": ["Actionable recommendations"],
  "strengths": ["Identified strengths"]
}
```

**Exit Codes:**
- `0` = PASS or CONDITIONAL
- `1` = FAIL
- `2` = ERROR

---

### 2. REP - Reasoning & Explanation Protocol

**Purpose**: Advanced multi-modal reasoning (deductive, inductive, abductive, analogical, causal)

**Usage:**
```bash
# Deductive reasoning
python run_rep_tool.py --reasoning-type deductive \
  --facts "Socrates is a man" "All men are mortal" \
  --rules '[["Socrates is a man", "All men are mortal"], "Socrates is mortal"]'

# From JSON file
python run_rep_tool.py --file reasoning_input.json
```

**Input (JSON):**
```json
{
  "reasoning_type": "deductive",
  "facts": ["fact1", "fact2"],
  "rules": [[["premise1"], "conclusion"]]
}
```

**Output:**
```json
{
  "reasoning_type": "deductive",
  "conclusions": ["Socrates is mortal"],
  "confidence_scores": {"Socrates is mortal": 0.95},
  "reasoning_chain": [...],
  "explanation": "Deductive inference: ..."
}
```

---

### 3. ESL - Emotional State Layer

**Purpose**: Multi-dimensional emotion detection and analysis

**Usage:**
```bash
# Analyze emotional content
python run_esl_tool.py --text "I'm absolutely thrilled about this!"

# Summary format
python run_esl_tool.py --text "..." --format summary
```

**Output:**
```json
{
  "emotional_state": "joy",
  "valence": "positive",
  "intensity": 0.9,
  "confidence": 0.92,
  "detected_emotions": [
    {
      "emotion": "joy",
      "intensity": 0.9,
      "dimension": "primary",
      "valence": "positive"
    }
  ]
}
```

---

### 4. VVP - Validation & Verification

**Purpose**: Input validation and logical structure verification

**Usage:**
```bash
# Validate rule structures
python run_vvp_tool.py --json '{"rules": [[["a"], "b"]]}'
```

**Output:**
```json
{
  "validation_status": "success",
  "reason": "Input validation passed: 1 rules checked"
}
```

---

## Example Scenarios

### Scenario 1: Paper2Agent Self-Governance

**Goal**: Validate own responses before returning to user

```python
#!/usr/bin/env python3
"""
Paper2Agent with Five Laws Self-Governance
"""
import subprocess
import json

def generate_response(user_query):
    """Generate response using your AI model"""
    # Your response generation logic
    return "Generated response based on query"

def validate_with_five_laws(response):
    """Validate response against Five Laws"""
    result = subprocess.run(
        ["python", "code/tools/run_five_laws_validator.py",
         "--text", response, "--strictness", "moderate"],
        capture_output=True,
        text=True
    )
    return json.loads(result.stdout)

def refine_response(original_response, recommendations):
    """Refine response based on recommendations"""
    # Your refinement logic based on recommendations
    return f"Refined: {original_response}"

def governed_response_pipeline(user_query, max_iterations=3):
    """Generate and validate response with governance"""

    for iteration in range(max_iterations):
        # Generate response
        response = generate_response(user_query)

        # Validate against Five Laws
        validation = validate_with_five_laws(response)

        # Check if compliant
        if validation["pass_fail_status"] == "PASS":
            print(f"✅ Governed response (Score: {validation['scores']['overall_compliance']:.1f}%)")
            return response

        # If not, refine based on recommendations
        print(f"⚠️  Iteration {iteration+1}: Refining based on {len(validation['recommendations'])} recommendations")
        response = refine_response(response, validation["recommendations"])

    # After max iterations, return best attempt with warning
    print("⚠️  Warning: Could not achieve PASS status within max iterations")
    return response

# Usage
if __name__ == "__main__":
    query = "Explain climate change"
    final_response = governed_response_pipeline(query)
    print(f"\nFinal Response: {final_response}")
```

---

### Scenario 2: Multi-Agent Validation

**Goal**: One agent validates another agent's output

```bash
#!/bin/bash

# Agent A generates response
agent_a_output=$(curl -s http://agent-a/generate?q="What is AI?")

# Agent B (Paper2Agent) validates using Five Laws
validation=$(echo "$agent_a_output" | python code/tools/run_five_laws_validator.py)

status=$(echo "$validation" | jq -r '.pass_fail_status')

if [ "$status" == "PASS" ]; then
    echo "✅ Agent A's response is governed"
    echo "$agent_a_output"
else
    echo "❌ Agent A's response failed governance"
    echo "$validation" | jq -r '.violations[]'

    # Send back for refinement
    curl -X POST http://agent-a/refine \
      -d "{\"original\": \"$agent_a_output\", \"recommendations\": $(echo "$validation" | jq '.recommendations')}"
fi
```

---

### Scenario 3: Governed Content Generation

**Goal**: Generate responses with built-in Five Laws compliance

```python
def generate_governed_response(prompt, target_compliance=85.0):
    """
    Generate response that meets Five Laws compliance threshold
    """
    import subprocess
    import json

    attempts = []

    for attempt in range(5):
        # Generate response (your logic here)
        response = your_llm_generate(prompt)

        # Validate
        result = subprocess.run(
            ["python", "code/tools/run_five_laws_validator.py", "--text", response],
            capture_output=True, text=True
        )
        validation = json.loads(result.stdout)

        compliance_score = validation["scores"]["overall_compliance"]
        attempts.append((response, compliance_score, validation))

        # Check if we met threshold
        if compliance_score >= target_compliance:
            print(f"✅ Generated governed response (Score: {compliance_score:.1f}%)")
            return response

        # Adjust prompt based on violations
        prompt = adjust_prompt_for_governance(prompt, validation["recommendations"])

    # Return best attempt
    best_response, best_score, best_validation = max(attempts, key=lambda x: x[1])
    print(f"⚠️  Best achieved: {best_score:.1f}% (target was {target_compliance}%)")
    return best_response
```

---

### Scenario 4: Continuous Integration / Quality Assurance

**Goal**: Validate AI-generated content in CI/CD pipeline

```yaml
# .github/workflows/ai_governance.yml
name: AI Content Governance Check

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r code/requirements.txt

      - name: Validate AI-generated content
        run: |
          for file in ai_generated_content/*.txt; do
            echo "Validating $file..."
            python code/tools/run_five_laws_validator.py --file "$file" --strictness strict

            if [ $? -ne 0 ]; then
              echo "❌ $file failed Five Laws governance"
              exit 1
            fi
          done

          echo "✅ All AI-generated content passes Five Laws governance"
```

---

## Protocol Composition

Tools can be chained together for complex governed workflows:

### Example 1: Reason → Validate → Govern

```bash
# Chain reasoning, validation, and governance
python run_rep_tool.py --json '{"reasoning_type": "deductive", ...}' | \
  python run_vvp_tool.py | \
  python run_five_laws_validator.py --format summary
```

### Example 2: Emotional Analysis + Governance

```bash
# Analyze emotion, then validate governance
python run_esl_tool.py --text "User message" > emotion_analysis.json

# Use emotion context in governance validation
python run_five_laws_validator.py --text "AI response" \
  --context "$(cat emotion_analysis.json)"
```

### Example 3: Multi-Step Governed Pipeline

```python
def governed_pipeline(user_input):
    """
    Complete governed processing pipeline
    """
    # Step 1: Analyze user emotion
    emotion = run_tool("run_esl_tool.py", {"text": user_input})

    # Step 2: Generate response with emotional awareness
    response = generate_empathetic_response(user_input, emotion)

    # Step 3: Apply reasoning validation
    reasoning = run_tool("run_rep_tool.py", {
        "reasoning_type": "deductive",
        "context": response
    })

    # Step 4: Validate structure
    validation = run_tool("run_vvp_tool.py", {"data": reasoning})

    # Step 5: Five Laws governance check
    governance = run_tool("run_five_laws_validator.py", {"text": response})

    if governance["pass_fail_status"] == "PASS":
        return response
    else:
        return refine_and_retry(response, governance["recommendations"])
```

---

## API Reference

### Five Laws Validator

**Command:**
```
python run_five_laws_validator.py [OPTIONS]
```

**Options:**
| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--text, -t` | string | Text to validate | - |
| `--file, -f` | path | File containing text | - |
| `--strictness, -s` | choice | lenient\|moderate\|strict | moderate |
| `--context, -c` | json | Additional context | - |
| `--format` | choice | json\|compact\|summary | json |
| `--verbose, -v` | flag | Enable verbose logging | false |

**Exit Codes:**
- `0`: PASS or CONDITIONAL
- `1`: FAIL
- `2`: ERROR

---

### REP - Reasoning Protocol

**Command:**
```
python run_rep_tool.py [OPTIONS]
```

**Options:**
| Flag | Type | Description |
|------|------|-------------|
| `--file, -f` | path | JSON file with reasoning input |
| `--json, -j` | json | JSON string with reasoning input |
| `--reasoning-type` | choice | deductive\|inductive\|abductive\|analogical\|causal |
| `--facts` | list | List of facts |
| `--rules` | json | JSON string of rules |

---

### ESL - Emotional State Layer

**Command:**
```
python run_esl_tool.py [OPTIONS]
```

**Options:**
| Flag | Type | Description | Default |
|------|------|-------------|---------|
| `--text, -t` | string | Text to analyze | - |
| `--file, -f` | path | File containing text | - |
| `--format` | choice | json\|summary | json |

---

### VVP - Validation Protocol

**Command:**
```
python run_vvp_tool.py [OPTIONS]
```

**Options:**
| Flag | Type | Description |
|------|------|-------------|
| `--file, -f` | path | JSON file with validation input |
| `--json, -j` | json | JSON string with validation input |

---

## Integration Patterns

### Pattern 1: Pre-Response Validation

```python
def send_response_with_governance(user_query):
    """Always validate before responding"""
    response = generate_response(user_query)

    validation = validate_five_laws(response)

    if validation["pass_fail_status"] != "PASS":
        response = apply_governance_fixes(response, validation)

    return response
```

### Pattern 2: Adaptive Strictness

```python
def adaptive_governance(response, context):
    """Adjust strictness based on context"""
    if context.get("domain") == "scientific":
        strictness = "strict"
    elif context.get("domain") == "creative":
        strictness = "lenient"
    else:
        strictness = "moderate"

    return validate_five_laws(response, strictness)
```

### Pattern 3: Iterative Refinement

```python
def iteratively_refine(initial_response, max_iterations=3):
    """Keep refining until governance passes"""
    response = initial_response

    for i in range(max_iterations):
        validation = validate_five_laws(response)

        if validation["pass_fail_status"] == "PASS":
            return response

        # Refine based on specific law violations
        response = refine_for_laws(response, validation)

    return response  # Best attempt
```

---

## Best Practices

### 1. Always Validate Critical Outputs

For responses involving facts, decisions, or user-facing content, always run Five Laws validation:

```python
if response_is_critical(response):
    validation = validate_five_laws(response, strictness="strict")
    if validation["pass_fail_status"] != "PASS":
        log_governance_failure(validation)
        response = fallback_response()
```

### 2. Use Appropriate Strictness

- **Strict**: Scientific content, legal text, medical information
- **Moderate**: General purpose responses, educational content
- **Lenient**: Creative writing, brainstorming, exploratory tasks

### 3. Log Governance Metrics

Track compliance over time:

```python
governance_metrics = {
    "timestamp": time.time(),
    "compliance_score": validation["scores"]["overall_compliance"],
    "status": validation["pass_fail_status"],
    "violations": len(validation["violations"])
}

log_to_analytics(governance_metrics)
```

### 4. Provide User Transparency

Optionally show governance scores to users:

```python
if show_governance_info:
    print(f"[Governed Response - Compliance: {score}%]")
    print(response)
```

---

## Troubleshooting

### Issue: Tool not found

**Solution:**
```bash
# Ensure you're in the correct directory
cd code/tools

# Or use full path
python /full/path/to/code/tools/run_five_laws_validator.py --text "..."
```

### Issue: Import errors

**Solution:**
```bash
# Install dependencies
cd code
pip install -r requirements.txt

# Verify imports
python -c "from tools.lib.five_laws_evaluator import evaluate_text; print('OK')"
```

### Issue: Low compliance scores

**Solution:**
Check specific law violations:
```bash
python run_five_laws_validator.py --text "..." --format summary
```

Review recommendations and refine response accordingly.

---

## Support & Resources

- **GitHub Repository**: https://github.com/dansasser/SIM-ONE
- **Tool Catalog**: [code/tools/README.md](code/tools/README.md)
- **Implementation Plan**: [docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md](docs/planning/PHASE_22_IMPLEMENTATION_PLAN.md)
- **Migration Plan**: [MIGRATION_PLAN.md](MIGRATION_PLAN.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## Next Steps

1. **Try the Five Laws Validator** with your AI system's responses
2. **Review the tools manifest** (`code/tools/tools_manifest.json`) for all available tools
3. **Experiment with protocol composition** to build governed workflows
4. **Integrate governance checks** into your AI pipeline
5. **Monitor compliance metrics** to track governance effectiveness

---

**Welcome to governed cognition. 🛡️**

*SIM-ONE Framework - Cognitive Governance for Autonomous AI Systems*
