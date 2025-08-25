import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class VVP:
    """
    A simple implementation of the Validation and Verification Protocol (VVP).
    This version validates the input data to allow for parallel execution.
    """

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the validation logic on the input data.

        Args:
            data: The initial data for the workflow.

        Returns:
            A dictionary with the validation status and a reason.
        """
        logger.info("Executing VVP on input data...")

        rules = data.get("rules")

        if not isinstance(rules, list):
            return {
                "validation_status": "failure",
                "reason": "Input validation failed: 'rules' field must be a list."
            }

        for i, rule in enumerate(rules):
            if not (isinstance(rule, list) and len(rule) == 2 and isinstance(rule[0], list)):
                 return {
                    "validation_status": "failure",
                    "reason": f"Input validation failed: Rule at index {i} is malformed."
                }

        reason = f"Input validation passed: {len(rules)} rules were checked and appear well-formed."
        logger.info(f"VVP: {reason}")

        return {
            "validation_status": "success",
            "reason": reason
        }

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    vvp_protocol = VVP()

    # --- Test Case 1: Successful Validation ---
    print("--- Test Case 1: Successful Validation ---")
    sample_data_success = {
        "facts": ["a", "b", "c"],
        "rules": [
            [["a"], "b"],
            [["b"], "c"]
        ]
    }
    result = vvp_protocol.execute(sample_data_success)
    print(result)
    assert result["validation_status"] == "success"

    # --- Test Case 2: Failed Validation ---
    print("\n--- Test Case 2: Failed Validation ---")
    sample_data_fail = {
        "facts": ["a", "b"],
        "rules": [
            "this is not a rule"
        ]
    }
    result = vvp_protocol.execute(sample_data_fail)
    print(result)
    assert result["validation_status"] == "failure"
