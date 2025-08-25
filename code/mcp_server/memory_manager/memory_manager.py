import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from mcp_server.config import settings
from mcp_server.database.memory_database import get_db_connection
from mcp_server.protocols.esl.esl import ESL

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages the storage and retrieval of memories, with a placeholder for semantic search.
    """
    def __init__(self):
        self.esl_protocol = ESL()

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

    def add_memories(self, session_id: str, memories: List[Dict[str, Any]]):
        # ... (same as before) ...
        if not memories: return
        for memory in memories:
            entity_name = memory.get("entity")
            if not entity_name: continue
            entity_id = self.get_or_create_entity(entity_name)
            if not entity_id: continue

            # Analyze emotional content with ESL Protocol
            esl_result = self.esl_protocol.execute({"user_input": memory.get("source_input", "")})

            # Prepare data for new schema
            self._execute_db_query(
                """
                INSERT INTO memories (
                    session_id, entity_id, content, emotional_state, source_protocol,
                    emotional_salience, rehearsal_count, last_accessed,
                    confidence_score, memory_type, actors, context_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    entity_id,
                    memory.get("source_input", ""),
                    json.dumps(esl_result), # Store the full ESL analysis
                    memory.get("source_protocol", "MTP"),
                    esl_result.get("salience", 0.5), # Use salience from ESL
                    memory.get("rehearsal_count", 0),
                    memory.get("last_accessed"),
                    esl_result.get("confidence", 1.0), # Use confidence from ESL
                    memory.get("memory_type", "episodic"),
                    json.dumps(memory.get("actors", [])),
                    json.dumps(memory.get("context_tags", {}))
                )
            )
        logger.info(f"Persisted {len(memories)} new memories to SQLite.")

    def get_all_memories(self, session_id: str) -> List[Dict[str, Any]]:
        # ... (same as before) ...
        query = "SELECT m.*, e.name as entity FROM memories m JOIN entities e ON m.entity_id = e.id WHERE m.session_id = ? ORDER BY m.timestamp DESC"
        rows = self._execute_db_query(query, (session_id,), fetch='all')
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

    def search_memories(self, session_id: str, query_text: str, top_k: int = 3, context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Performs a mock semantic search using simple keyword matching.
        """
        logger.info(f"Performing mock semantic search for: '{query_text}' in session {session_id} with context {context}")
        all_memories = self.get_all_memories(session_id)

        query_words = set(query_text.lower().split())

        scored_memories = []
        for mem in all_memories:
            # 1. Keyword Score
            content_words = set(mem.get("content", "").lower().split())
            keyword_score = len(query_words.intersection(content_words))

            # 2. Emotional Salience Boost
            emotional_salience = mem.get("emotional_salience", 0.5)
            salience_boost = 1 + emotional_salience

            # 3. Rehearsal Boost
            rehearsal_boost = mem.get("rehearsal_count", 0) * 0.1

            # 4. Recency Boost (simple implementation)
            recency_boost = 0
            if mem.get("last_accessed"):
                # A more complex implementation would calculate time decay
                recency_boost = 0.2

            # 5. Actor Boost
            actor_boost = 0
            if context and 'actors' in context:
                memory_actors = mem.get('actors', [])
                if any(actor in memory_actors for actor in context['actors']):
                    actor_boost = 1.0 # Significant boost for matching actors

            # Combine scores
            final_score = (keyword_score * salience_boost) + rehearsal_boost + recency_boost + actor_boost

            if final_score > 0:
                scored_memories.append({"score": final_score, "memory": mem})

        scored_memories.sort(key=lambda x: x['score'], reverse=True)

        retrieved_memories = [item['memory'] for item in scored_memories[:top_k]]

        # Update rehearsal and recency for retrieved memories
        for mem in retrieved_memories:
            self._execute_db_query(
                "UPDATE memories SET rehearsal_count = rehearsal_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                (mem['id'],)
            )

        return retrieved_memories

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
