import logging
import re
import json
import sys
import os
from typing import Dict, Any, List

# Allow running this script directly for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from mcp_server.memory_manager.memory_manager import MemoryManager
from mcp_server.protocols.mtp import entity_patterns

logger = logging.getLogger(__name__)

class MTP:
    """
    An advanced Memory Tagger Protocol (MTP) that performs sophisticated, rule-based
    entity extraction, relationship detection, and emotional tagging.
    """
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.entity_patterns = entity_patterns.ENTITY_PATTERNS
        self.relationship_patterns = entity_patterns.RELATIONSHIP_PATTERNS
        self.entity_processing_order = ['organization', 'place', 'person', 'event', 'object', 'concept']

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extracts entities using regex patterns and prevents overlapping matches."""
        found_spans = {}
        for entity_type in self.entity_processing_order:
            patterns = self.entity_patterns.get(entity_type, [])
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = match.span()
                    # Skip if this span is already contained within another
                    if any(s[0] <= span[0] and s[1] >= span[1] and s != span for s in found_spans):
                        continue

                    # If this exact span is already found, the first type (higher priority) wins
                    if span in found_spans:
                        continue

                    entity_name = match.group(1) if match.groups() else match.group(0)
                    found_spans[span] = {"entity": entity_name.strip(), "type": entity_type, "span": span}

        return sorted(list(found_spans.values()), key=lambda x: x['span'][0])

    def _detect_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Finds relationship keywords and connects the nearest valid source/target entities."""
        relationships = []
        person_centric_rels = ['works_at', 'located_in']

        for rel_type, pattern in self.relationship_patterns.items():
            for rel_match in pattern.finditer(text):
                rel_span = rel_match.span()

                source_entity, target_entity = None, None

                # Find nearest entity before the relationship
                candidates_before = [e for e in entities if e['span'][1] <= rel_span[0]]
                # If it's a person-centric relationship, strongly prefer a person as the source
                if rel_type in person_centric_rels:
                    person_candidates = [p for p in candidates_before if p['type'] == 'person']
                    if person_candidates:
                        candidates_before = person_candidates # Prioritize persons

                if candidates_before:
                    source_entity = min(candidates_before, key=lambda e: rel_span[0] - e['span'][1])

                # Find nearest entity after the relationship
                candidates_after = [e for e in entities if e['span'][0] >= rel_span[1]]
                if candidates_after:
                    target_entity = min(candidates_after, key=lambda e: e['span'][0] - rel_span[1])

                if source_entity and target_entity:
                    is_valid = False
                    if rel_type == 'works_at' and source_entity['type'] == 'person' and target_entity['type'] == 'organization':
                        is_valid = True
                    elif rel_type == 'located_in' and source_entity['type'] == 'person' and target_entity['type'] == 'place':
                        is_valid = True

                    if is_valid:
                        rel = {
                            "source": source_entity['entity'], "target": target_entity['entity'],
                            "relationship_type": rel_type, "strength": 0.9
                        }
                        if rel not in relationships:
                            relationships.append(rel)
                            logger.info(f"Found relationship: {source_entity['entity']} -> {rel_type} -> {target_entity['entity']}")
        return relationships

    def _calculate_salience(self, entity: Dict, text: str) -> float:
        salience = 0.5
        if entity['type'] in ['person', 'organization', 'place']: salience += 0.2
        salience += 0.1 * text.lower().count(entity['entity'].lower())
        return min(1.0, round(salience, 2))

    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        user_input, emotional_context, session_id = data.get("user_input", ""), data.get("emotional_context", {}), data.get("session_id")
        if not user_input or not session_id: return {"status": "skipped", "reason": "Missing user_input or session_id."}

        extracted_entities = self._extract_entities(user_input)
        entity_relationships = self._detect_relationships(user_input, extracted_entities)

        processed_entities = [
            {
                "entity": e['entity'], "type": e['type'], "salience": self._calculate_salience(e, user_input),
                "emotional_state": emotional_context.get('valence', 'neutral'),
                "context": user_input[max(0, e['span'][0]-20):e['span'][1]+20],
                "relationships": list(set([r['target'] for r in entity_relationships if r['source'] == e['entity']])),
                "confidence": 0.85, "first_mention": user_input.find(e['entity']) == e['span'][0]
            } for e in extracted_entities
        ]

        memory_tags = self._create_memory_tags(processed_entities, user_input, emotional_context)
        if memory_tags: self.memory_manager.add_memories(session_id, memory_tags)

        return {
            "extracted_entities": processed_entities, "entity_relationships": entity_relationships,
            "memory_tags": memory_tags,
            "contextual_factors": {"entity_count": len(processed_entities), "new_entities": len(memory_tags)},
            "explanation": f"Extracted {len(processed_entities)} entities. Found {len(entity_relationships)} relationships."
        }

    def _create_memory_tags(self, entities: List[Dict], user_input: str, emotional_context: Dict) -> List[Dict]:
        return [
            {
                "entity": e['entity'],
                "emotions": json.dumps(emotional_context.get('detected_emotions', [])),
                "emotional_state": emotional_context.get('valence', 'neutral'),
                "source_input": user_input,
                "emotional_salience": e['salience'],
                "source_protocol": "MTP"
            } for e in entities
        ]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    mtp = MTP()
    mock_esl_output = {"valence": "positive", "detected_emotions": [{"emotion": "joy", "intensity": 0.8}]}
    test_input = "John works at Microsoft and lives in Seattle."
    print(f"--- Testing Advanced MTP (v4) ---")
    print(f"Input: '{test_input}'")

    class MockMemoryManager:
        def add_memories(self, session_id, memories):
            print(f"\n--- Mock MemoryManager: Adding {len(memories)} memories to session {session_id} ---")
            for mem in memories: print(f"  - {mem}")
    mtp.memory_manager = MockMemoryManager()

    result = mtp.execute({"user_input": test_input, "emotional_context": mock_esl_output, "session_id": "test-session-123"})
    print("\n--- MTP Execution Result ---")
    print(json.dumps(result, indent=2))

    extracted_entities_map = {e['entity']: e['type'] for e in result['extracted_entities']}
    assert extracted_entities_map.get("John") == "person", f"John was {extracted_entities_map.get('John')}"
    assert extracted_entities_map.get("Microsoft") == "organization", f"Microsoft was {extracted_entities_map.get('Microsoft')}"
    assert extracted_entities_map.get("Seattle") == "place", f"Seattle was {extracted_entities_map.get('Seattle')}"
    assert len(result['entity_relationships']) == 2, f"Found {len(result['entity_relationships'])} relationships"
    rel_tuples = {(r['source'], r['relationship_type'], r['target']) for r in result['entity_relationships']}
    assert ("John", "works_at", "Microsoft") in rel_tuples
    assert ("John", "located_in", "Seattle") in rel_tuples
    print("\n--- All Assertions Passed! ---")
