import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.protocols.rep.rep import AdvancedREP

def test_deductive_reasoning():
    rep = AdvancedREP()
    test_data = {
        "reasoning_type": "deductive",
        "facts": ["Socrates is a man", "All men are mortal"],
        "rules": [ (["Socrates is a man", "All men are mortal"], "Socrates is mortal"),
        ]
    }
    result = rep.execute(test_data)
    assert "Socrates is mortal" in result['conclusions']
    assert result['validation']['is_valid'] == True
    assert len(result['reasoning_chain']) > 0
    print(" Deductive reasoning test passed")

def test_inductive_reasoning():
    rep = AdvancedREP()
    test_data = {
        "reasoning_type": "inductive",
        "observations": [
            "The sun rose in the east today",
            "The sun rose in the east yesterday",
            "The sun rose in the east last week"
        ]
    }
    result = rep.execute(test_data)
    assert len(result['conclusions']) > 0
    assert result['validation']['is_valid'] == True
    print(" Inductive reasoning test passed")

def test_confidence_scoring():
    rep = AdvancedREP()
    test_data = {
        "reasoning_type": "deductive",
        "facts": ["A is true"],
        "rules": [
            (["A is true"], "B is true"),
            (["B is true"], "C is true")
        ]
    }
    result = rep.execute(test_data)
    # Confidence should degrade through chain
    confidences = list(result['confidence_scores'].values())
    assert all(0 < conf <= 1 for conf in confidences)
    print(" Confidence scoring test passed")

if __name__ == '__main__':
    test_deductive_reasoning()
    test_inductive_reasoning()
    test_confidence_scoring()
    print(" All tests passed!")
