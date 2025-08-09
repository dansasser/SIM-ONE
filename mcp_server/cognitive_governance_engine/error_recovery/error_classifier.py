from typing import Dict, Any

def classify_error(error: Exception, protocol_name: str) -> Dict[str, Any]:
    """
    Classifies a given exception into a structured error format.

    Args:
        error: The exception object that was raised.
        protocol_name: The name of the protocol where the error occurred.

    Returns:
        A dictionary containing the classified error information.
    """
    error_type = type(error).__name__
    severity = "low"  # Default severity

    # Heuristics for determining severity
    if isinstance(error, (ConnectionError, TimeoutError)):
        severity = "high"
    elif isinstance(error, (TypeError, ValueError, KeyError, AttributeError)):
        severity = "medium"
    elif isinstance(error, NotImplementedError):
        severity = "high"

    return {
        "type": error_type,
        "severity": severity,
        "protocol": protocol_name,
        "message": str(error)
    }

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Error Classifier ---")

    # Test Case 1: ConnectionError (High Severity)
    try:
        raise ConnectionError("Failed to connect to service.")
    except ConnectionError as e:
        classified_error1 = classify_error(e, "REP")
        print(f"Test 1 (ConnectionError): {classified_error1}")
        assert classified_error1['type'] == 'ConnectionError'
        assert classified_error1['severity'] == 'high'
        assert classified_error1['protocol'] == 'REP'

    # Test Case 2: ValueError (Medium Severity)
    try:
        raise ValueError("Invalid input provided.")
    except ValueError as e:
        classified_error2 = classify_error(e, "ESL")
        print(f"Test 2 (ValueError): {classified_error2}")
        assert classified_error2['type'] == 'ValueError'
        assert classified_error2['severity'] == 'medium'

    # Test Case 3: A generic Exception (Low Severity)
    try:
        raise Exception("A generic error.")
    except Exception as e:
        classified_error3 = classify_error(e, "MTP")
        print(f"Test 3 (Generic Exception): {classified_error3}")
        assert classified_error3['type'] == 'Exception'
        assert classified_error3['severity'] == 'low'
