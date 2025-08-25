import re
from fastapi import HTTPException

# Basic patterns to detect common injection attacks.
# In a real-world application, these would be much more comprehensive
# and you would likely use a library like `py-trie-sql-parser` for SQLi.
SQLI_PATTERNS = [
    r"(\s*')\s*OR\s+'\d+'\s*=\s*'\d+'",
    r"(\s*)\b(UNION|SELECT|INSERT|UPDATE|DELETE|DROP|ALTER)\b(\s*)",
    r"(\s*)--",
    r"(\s*);",
]

CMD_INJECTION_PATTERNS = [
    r";\s*(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)",
    r"&&\s*(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)",
    r"`(ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)`",
    r"\$\((ls|dir|cat|whoami|pwd|sh|bash|powershell|cmd)\)",
]

XSS_PATTERNS = [
    r"<script.*?>.*?</script>",
    r"javascript:",
    r"onerror=",
    r"onload=",
    r"<iframe.*?>",
]

def check_for_patterns(text: str, patterns: list[str], error_message: str):
    """
    Checks a string against a list of regex patterns and raises an HTTPException if a match is found.
    """
    if not isinstance(text, str):
        return # Only check strings

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise HTTPException(status_code=400, detail=f"Malicious input detected: {error_message}")

def advanced_validate_input(data: any):
    """
    Recursively traverses a data structure (dict, list, string) and checks all string values
    against the defined injection patterns.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            advanced_validate_input(v)
    elif isinstance(data, list):
        for item in data:
            advanced_validate_input(item)
    elif isinstance(data, str):
        check_for_patterns(data, SQLI_PATTERNS, "Potential SQL Injection")
        check_for_patterns(data, CMD_INJECTION_PATTERNS, "Potential Command Injection")
        check_for_patterns(data, XSS_PATTERNS, "Potential XSS")
