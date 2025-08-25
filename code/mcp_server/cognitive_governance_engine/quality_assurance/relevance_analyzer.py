import sys
import os
from typing import Dict, Any

# Adjust path to import from sibling directory 'coherence_validator'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from coherence_validator.semantic_analyzer import get_keywords

def analyze_relevance(protocol_output: Dict[str, Any], user_input: str) -> Dict[str, Any]:
    """
    Analyzes the relevance of a protocol's output to the original user input.
    Relevance is simulated by checking for keyword overlap.

    Returns:
        A dictionary with a relevance score and a reason.
    """
    if not user_input:
        return {"relevant": True, "score": 1.0, "reason": "No user input to compare against."}

    input_keywords = get_keywords(user_input)

    # Aggregate all text from the protocol output to create a set of output keywords
    output_text = ""
    if isinstance(protocol_output, dict):
        # A simple way to get all text values from a dictionary
        for value in protocol_output.values():
            if isinstance(value, str):
                output_text += value + " "
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        output_text += item + " "
                    # Handle dicts inside lists, like MTP's entities
                    elif isinstance(item, dict) and 'entity' in item:
                        output_text += item['entity'] + " "

    output_keywords = get_keywords(output_text)

    if not input_keywords or not output_keywords:
        # If either set is empty, can't determine relevance, so assume it's fine.
        return {"relevant": True, "score": 1.0, "reason": "Not enough keywords for relevance analysis."}

    # Calculate overlap
    intersection = input_keywords.intersection(output_keywords)

    # Relevance score is the ratio of shared keywords to the total number of output keywords.
    # This measures how much of the output is "about" the input.
    relevance_score = len(intersection) / len(output_keywords)

    if relevance_score < 0.3:
        return {
            "relevant": False,
            "score": round(relevance_score, 2),
            "reason": f"Output has low keyword overlap with the input, suggesting potential irrelevance."
        }

    return {"relevant": True, "score": round(relevance_score, 2), "reason": "Output keywords are relevant to the input."}

if __name__ == '__main__':
    # Example Usage
    print("--- Testing Relevance Analyzer ---")

    # Test Case 1: Relevant output
    inp1 = "The new phone from Apple is amazing."
    out1 = {"conclusion": "The Apple phone is a high-quality product."}
    relevance1 = analyze_relevance(out1, inp1)
    print(f"Relevance 1 (Relevant): {relevance1}")
    assert relevance1['relevant'] is True
    assert relevance1['score'] > 0.5

    # Test Case 2: Irrelevant output
    inp2 = "What is the status of the server migration?"
    out2 = {"conclusion": "The sky is blue."}
    relevance2 = analyze_relevance(out2, inp2)
    print(f"Relevance 2 (Irrelevant): {relevance2}")
    assert relevance2['relevant'] is False
    assert relevance2['score'] < 0.3

    # Test Case 3: MTP style output
    inp3 = "John Doe works for Microsoft."
    out3 = {"extracted_entities": [{"entity": "John Doe"}, {"entity": "Microsoft"}]}
    relevance3 = analyze_relevance(out3, inp3)
    print(f"Relevance 3 (MTP Relevant): {relevance3}")
    assert relevance3['relevant'] is True
    assert relevance3['score'] == 1.0
