import logging
import difflib
import re
from datetime import datetime, timedelta

from mcp_server.database.memory_database import get_db_connection
from mcp_server.memory_manager.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class MemoryConsolidationEngine:
    """
    Handles background tasks for memory maintenance, including merging,
    contradiction detection, and archiving.
    """
    def __init__(self):
        self.memory_manager = MemoryManager()

    def _find_similar_memories(self, session_id: str, similarity_threshold=0.85):
        """Finds groups of memories with similar content to merge."""
        logger.info(f"Consolidation: Finding similar memories for session {session_id}...")
        memories = self.memory_manager.get_all_memories(session_id)

        # O(n^2) - not efficient for large memory stores
        groups = []
        processed_ids = set()
        for i in range(len(memories)):
            if memories[i]['id'] in processed_ids:
                continue

            current_group = [memories[i]]
            for j in range(i + 1, len(memories)):
                if memories[j]['id'] in processed_ids:
                    continue

                ratio = difflib.SequenceMatcher(None, memories[i]['content'], memories[j]['content']).ratio()
                if ratio > similarity_threshold:
                    current_group.append(memories[j])

            if len(current_group) > 1:
                groups.append(current_group)
                for mem in current_group:
                    processed_ids.add(mem['id'])

        return groups

    def _merge_memories(self, session_id: str, memory_group: list):
        """Merges a group of similar memories into a single, more robust memory."""
        if not memory_group:
            return

        logger.info(f"Consolidation: Merging {len(memory_group)} memories...")

        # Create a new merged memory
        # For V1, we'll just take the content of the most recent memory
        memory_group.sort(key=lambda m: m['timestamp'], reverse=True)
        merged_content = memory_group[0]['content']

        # Combine metadata
        new_rehearsal_count = sum(m.get('rehearsal_count', 0) for m in memory_group)
        new_emotional_salience = max(m.get('emotional_salience', 0.5) for m in memory_group)
        new_actors = list(set(actor for mem in memory_group for actor in mem.get('actors', [])))

        # Add the new merged memory
        new_memory = {
            "entity": memory_group[0]['entity'],
            "source_input": merged_content,
            "rehearsal_count": new_rehearsal_count,
            "emotional_salience": new_emotional_salience,
            "actors": new_actors,
            "memory_type": "semantic" # Merged memories become semantic
        }
        self.memory_manager.add_memories(session_id, [new_memory])

        # Archive the old memories
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                for mem in memory_group:
                    cursor.execute("UPDATE memories SET memory_type = 'archived' WHERE id = ?", (mem['id'],))
                conn.commit()
            finally:
                conn.close()

    def _find_contradictions(self, session_id: str):
        """
        A simple contradiction finder. Looks for simple negations.
        e.g., "x is good" vs "x is not good"
        """
        logger.info(f"Consolidation: Finding contradictions for session {session_id}...")
        memories = self.memory_manager.get_all_memories(session_id)
        contradictions = []

        # This is a very naive O(n^2) implementation. A real system would need something more efficient.
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                mem1 = memories[i]
                mem2 = memories[j]

                # Simple negation check
                negation_pattern = r"\b(not|n't|never)\b"
                mem1_has_negation = re.search(negation_pattern, mem1['content'], re.IGNORECASE)
                mem2_has_negation = re.search(negation_pattern, mem2['content'], re.IGNORECASE)

                # XOR condition: one must have negation and the other must not.
                if bool(mem1_has_negation) != bool(mem2_has_negation):
                    if mem1_has_negation:
                        base_content = re.sub(negation_pattern, "", mem1['content'], 1).strip()
                        comparison_content = mem2['content']
                    else:
                        base_content = re.sub(negation_pattern, "", mem2['content'], 1).strip()
                        comparison_content = mem1['content']

                    if difflib.SequenceMatcher(None, base_content, comparison_content).ratio() > 0.8:
                        contradictions.append((mem1['id'], mem2['id'], "Simple negation detected"))

        return contradictions

    def _archive_old_memories(self, session_id: str):
        """
        Archives old, low-salience, unrehearsed memories.
        """
        logger.info(f"Consolidation: Archiving old memories for session {session_id}...")

        archive_threshold_days = 30
        max_rehearsal_count = 2
        max_salience = 0.4

        archive_date = datetime.now() - timedelta(days=archive_threshold_days)

        query = """
            UPDATE memories
            SET memory_type = 'archived'
            WHERE session_id = ?
              AND memory_type != 'archived'
              AND rehearsal_count <= ?
              AND emotional_salience < ?
              AND timestamp < ?
        """

        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(query, (session_id, max_rehearsal_count, max_salience, archive_date.strftime('%Y-%m-%d %H:%M:%S')))
                conn.commit()
                logger.info(f"Archived {cursor.rowcount} old memories for session {session_id}.")
            finally:
                conn.close()

    def run_consolidation_cycle(self, session_id: str):
        """
        Runs a full consolidation cycle for a given session.
        """
        logger.info(f"--- Starting memory consolidation cycle for session {session_id} ---")

        # 1. Merge similar memories
        similar_groups = self._find_similar_memories(session_id)
        for group in similar_groups:
            self._merge_memories(session_id, group)

        # 2. Find and flag contradictions
        contradictions = self._find_contradictions(session_id)
        if contradictions:
            conn = get_db_connection()
            if conn:
                try:
                    cursor = conn.cursor()
                    for mem1_id, mem2_id, reason in contradictions:
                        # Avoid duplicate entries
                        cursor.execute("SELECT id FROM memory_contradictions WHERE (memory_id_1 = ? AND memory_id_2 = ?) OR (memory_id_1 = ? AND memory_id_2 = ?)", (mem1_id, mem2_id, mem2_id, mem1_id))
                        if cursor.fetchone() is None:
                            cursor.execute("INSERT INTO memory_contradictions (memory_id_1, memory_id_2, reason) VALUES (?, ?, ?)", (mem1_id, mem2_id, reason))
                    conn.commit()
                    logger.info(f"Flagged {len(contradictions)} new contradictions for session {session_id}.")
                finally:
                    conn.close()

        # 3. Archive old memories
        self._archive_old_memories(session_id)

        logger.info(f"--- Finished memory consolidation cycle for session {session_id} ---")

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    engine = MemoryConsolidationEngine()
    # In a real scenario, you would iterate through active sessions
    engine.run_consolidation_cycle("test-session")
