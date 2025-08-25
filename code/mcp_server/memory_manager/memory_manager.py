import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from mcp_server.config import settings
from mcp_server.database.memory_database import get_db_connection

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the storage and retrieval of memories, with a placeholder for semantic search.
    """

    def _execute_db_query(self, query: str, params: tuple = (), fetch: str = None):
        # ... (same as before) ...
        conn = get_db_connection()
        if not conn: return None
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            if fetch == 'one': result = cursor.fetchone()
            elif fetch == 'all': result = cursor.fetchall()
            else: result = None
            conn.commit()
            return result
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            return None
        finally:
            if conn: conn.close()

    def get_or_create_entity(self, entity_name: str, entity_type: str = "unknown") -> Optional[int]:
        # ... (same as before) ...
        entity = self._execute_db_query("SELECT id FROM entities WHERE name = ?", (entity_name,), fetch='one')
        if entity: return entity['id']
        else:
            self._execute_db_query("INSERT INTO entities (name, type) VALUES (?, ?)", (entity_name, entity_type))
            new_entity = self._execute_db_query("SELECT id FROM entities WHERE name = ?", (entity_name,), fetch='one')
            return new_entity['id'] if new_entity else None

    def add_memories(self, memories: List[Dict[str, Any]]):
        # ... (same as before) ...
        if not memories: return
        for memory in memories:
            entity_name = memory.get("entity")
            if not entity_name: continue
            entity_id = self.get_or_create_entity(entity_name)
            if not entity_id: continue

            # Prepare data for new schema, using defaults for now
            self._execute_db_query(
                """
                INSERT INTO memories (
                    entity_id, content, emotional_state, source_protocol,
                    emotional_salience, rehearsal_count, last_accessed,
                    confidence_score, memory_type, actors, context_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_id,
                    memory.get("source_input", ""),
                    memory.get("emotional_state"),
                    memory.get("source_protocol", "MTP"),
                    memory.get("emotional_salience", 0.5),
                    memory.get("rehearsal_count", 0),
                    memory.get("last_accessed"),
                    memory.get("confidence_score", 1.0),
                    memory.get("memory_type", "episodic"),
                    json.dumps(memory.get("actors", [])),
                    json.dumps(memory.get("context_tags", {}))
                )
            )
        logger.info(f"Persisted {len(memories)} new memories to SQLite.")

    def get_all_memories(self) -> List[Dict[str, Any]]:
        # ... (same as before) ...
        rows = self._execute_db_query("SELECT m.*, e.name as entity FROM memories m JOIN entities e ON m.entity_id = e.id ORDER BY m.timestamp DESC", fetch='all')
        if not rows: return []

        memories = []
        for row in rows:
            memory = dict(row)
            # Deserialize JSON fields
            if memory.get('actors'):
                memory['actors'] = json.loads(memory['actors'])
            if memory.get('context_tags'):
                memory['context_tags'] = json.loads(memory['context_tags'])
            memories.append(memory)
        return memories

    def search_memories(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Performs a mock semantic search using simple keyword matching.
        """
        logger.info(f"Performing mock semantic search for: '{query_text}'")
        all_memories = self.get_all_memories()

        query_words = set(query_text.lower().split())

        scored_memories = []
        for mem in all_memories:
            content_words = set(mem.get("content", "").lower().split())
            score = len(query_words.intersection(content_words))
            if score > 0:
                scored_memories.append({"score": score, "memory": mem})

        scored_memories.sort(key=lambda x: x['score'], reverse=True)

        return [item['memory'] for item in scored_memories[:top_k]]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    db_file = Path(__file__).parent.parent / "database/persistent_memory.db"
    if os.path.exists(db_file): os.remove(db_file)
    from mcp_server.database.memory_database import initialize_database
    initialize_database()

    mm = MemoryManager()
    print("--- Testing Mock Semantic Search ---")

    memories_to_add = [
        {"entity": "AI Safety", "source_input": "AI safety is a critical field of research."},
        {"entity": "Jules", "source_input": "Jules works on AI."},
        {"entity": "Governance", "source_input": "AI governance requires careful thought."}
    ]
    mm.add_memories(memories_to_add)

    search_results = mm.search_memories("What do you know about AI safety and governance?")
    print("\nSearch Results:")
    print(json.dumps(search_results, indent=2))
    assert len(search_results) == 2
    # FIX: Correct assertion to check for substring presence
    assert any("safety" in m['content'] for m in search_results)
    assert any("governance" in m['content'] for m in search_results)

    print("\nTest finished successfully.")
