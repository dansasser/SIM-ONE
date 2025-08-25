import re

# This file contains the regex patterns for the MTP protocol's entity extraction.

# General pattern for capitalized words/phrases, which often signify named entities.
PROPER_NOUN_PATTERN = re.compile(r"\b([A-Z][a-z']+(?:\s+[A-Z][a-z']+)*)\b")

# Patterns for specific entity types.
# These are a mix of keywords and structural patterns.
ENTITY_PATTERNS = {
    "person": [
        # Titles followed by a proper noun (e.g., "Mr. Smith", "CEO John Doe")
        re.compile(r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|CEO|CFO|CTO)\s+([A-Z][a-z']+(?:\s+[A-Z][a-z']+)*)\b"),
        # A simple proper noun pattern, will be used carefully to avoid false positives
        PROPER_NOUN_PATTERN
    ],
    "place": [
        # Specific location keywords that are less ambiguous
        re.compile(r"\b(Seattle|downtown|New York|California)\b", re.IGNORECASE)
    ],
    "organization": [
        # Proper nouns followed by a company suffix
        re.compile(r"\b([A-Z][a-z']+(?:\s+[A-Z][a-z']+)*)\s+(?:Inc\.|Corp\.|LLC|Ltd\.)\b"),
        # Known company names, to give them precedence
        re.compile(r"\b(Microsoft|OpenAI|Google)\b", re.IGNORECASE)
    ],
    "event": [
        re.compile(r"\b(meeting|appointment|call|conference|deadline)\b", re.IGNORECASE),
        re.compile(r"\b(tomorrow|today|yesterday|next week)\b", re.IGNORECASE)
    ],
    "object": [
        re.compile(r"\b(iPhone|phone|computer|document)\b", re.IGNORECASE)
    ],
    "concept": [
        re.compile(r"\b(project|idea|plan|system|stress|quality)\b", re.IGNORECASE)
    ]
}

# Patterns to identify relationships between entities.
RELATIONSHIP_PATTERNS = {
    "works_at": re.compile(r"\s(works at|employed by|works for)\s", re.IGNORECASE),
    "located_in": re.compile(r"\s(in|located in|based in|lives in)\s", re.IGNORECASE),
    "related_to": re.compile(r"\s(friend of|colleague of|related to|connected to)\s", re.IGNORECASE),
    "meeting_with": re.compile(r"\s(meeting with|call with)\s", re.IGNORECASE)
}

# Pronouns are removed from automatic extraction for now to simplify logic.
# A proper pronoun resolution step would be needed for them.
PRONOUNS = {
    "person": re.compile(r"\b(he|she|him|her|they|them|my friend)\b", re.IGNORECASE),
    "object": re.compile(r"\b(it|this|that)\b", re.IGNORECASE)
}
