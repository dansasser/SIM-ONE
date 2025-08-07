import logging
from typing import List, Set, Tuple, Dict, Any

logger = logging.getLogger(__name__)

class REP:
    """
    A simple implementation of the Reasoning and Explanation Protocol (REP)
    using a rule-based forward-chaining inference engine.
    """

    def execute(self, facts: List[str], rules: List[Tuple[List[str], str]]) -> Dict[str, Any]:
        """
        Executes the reasoning engine.

        Args:
            facts: A list of initial facts.
            rules: A list of rules, where each rule is a tuple of (premises, conclusion).

        Returns:
            A dictionary with the conclusions and the explanation.
        """
        known_facts: Set[str] = set(facts)
        explanation: List[str] = [f"Initial facts: {known_facts}"]

        logger.info(f"Executing REP with {len(facts)} facts and {len(rules)} rules.")

        new_fact_derived = True
        while new_fact_derived:
            new_fact_derived = False
            for premises, conclusion in rules:
                if conclusion not in known_facts:
                    if all(premise in known_facts for premise in premises):
                        known_facts.add(conclusion)
                        explanation_step = f"Rule applied: IF {' AND '.join(premises)} THEN {conclusion}. New fact derived: {conclusion}"
                        explanation.append(explanation_step)
                        logger.debug(explanation_step)
                        new_fact_derived = True

        derived_facts = known_facts - set(facts)
        logger.info(f"REP execution finished. Derived {len(derived_facts)} new facts.")

        return {
            "conclusions": list(derived_facts),
            "explanation": explanation
        }

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO)

    initial_facts = ["Socrates is a man", "All men are mortal"]
    knowledge_rules = [
        (["Socrates is a man", "All men are mortal"], "Socrates is mortal"),
    ]

    rep_protocol = REP()
    result = rep_protocol.execute(initial_facts, knowledge_rules)

    logger.info("Conclusions: %s", result["conclusions"])
    logger.info("Explanation:")
    for step in result["explanation"]:
        logger.info("- %s", step)

    # A more complex example
    logger.info("\n--- More Complex Example ---")
    facts = ["has_feathers", "flies", "lays_eggs"]
    rules = [
        (["has_feathers"], "is_bird"),
        (["flies", "is_bird"], "is_flying_bird"),
        (["is_bird", "lays_eggs"], "is_oviparous_bird")
    ]
    result = rep_protocol.execute(facts, rules)
    logger.info("Conclusions: %s", result["conclusions"])
    logger.info("Explanation:")
    for step in result["explanation"]:
        logger.info("- %s", step)
