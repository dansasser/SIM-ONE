import psutil
import logging

from mcp_server.database.memory_database import get_db_connection

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    A class to collect various system and application metrics.
    """

    def get_system_metrics(self) -> dict:
        """Collects basic system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }

    def get_application_metrics(self) -> dict:
        """Collects application-specific metrics."""
        # For V1, we will focus on memory counts.
        # This can be expanded to track protocol usage, API errors, etc.
        db_conn = get_db_connection()
        if not db_conn:
            logger.warning("MetricsCollector: Could not connect to database to get memory stats.")
            return {"memory_stats": "unavailable"}

        try:
            cursor = db_conn.cursor()

            # Total memories
            cursor.execute("SELECT COUNT(*) FROM memories")
            total_memories = cursor.fetchone()[0]

            # Memories by type
            cursor.execute("SELECT memory_type, COUNT(*) FROM memories GROUP BY memory_type")
            memories_by_type = dict(cursor.fetchall())

            # Total contradictions
            cursor.execute("SELECT COUNT(*) FROM memory_contradictions")
            total_contradictions = cursor.fetchone()[0]

            return {
                "memory_stats": {
                    "total_memories": total_memories,
                    "by_type": memories_by_type,
                    "total_contradictions": total_contradictions,
                }
            }
        except Exception as e:
            logger.error(f"MetricsCollector: Failed to get application metrics: {e}")
            return {"memory_stats": "error"}
        finally:
            if db_conn:
                db_conn.close()

    def get_all_metrics(self) -> dict:
        """Returns a combined dictionary of all metrics."""
        metrics = {
            "system": self.get_system_metrics(),
            "application": self.get_application_metrics(),
        }
        return metrics
