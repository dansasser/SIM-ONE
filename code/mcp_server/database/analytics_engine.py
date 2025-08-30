import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, Counter
import statistics
import json

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

@dataclass
class QueryPattern:
    """Represents a query pattern for analysis."""
    pattern_hash: str
    normalized_query: str
    query_type: str
    table_names: List[str]
    execution_count: int
    avg_duration: float
    min_duration: float
    max_duration: float
    total_duration: float
    error_count: int
    last_execution: datetime
    first_seen: datetime

@dataclass
class MemoryAnalytics:
    """Analytics data for memory operations."""
    total_memories: int
    memories_by_entity: Dict[str, int]
    memories_by_session: Dict[str, int]
    memories_by_protocol: Dict[str, int]
    emotional_salience_distribution: Dict[str, int]
    memory_type_distribution: Dict[str, int]
    average_memory_length: float
    recent_activity: List[Dict[str, Any]]

@dataclass
class PerformanceMetrics:
    """Database performance metrics."""
    queries_per_second: float
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    throughput_trend: List[Tuple[datetime, float]]
    slow_query_count: int
    connection_utilization: float

class DatabaseAnalyticsEngine:
    """
    Comprehensive database analytics and performance tracking engine.
    Provides insights into query patterns, memory usage, and system performance.
    """
    
    def __init__(self):
        self.analytics_cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.slow_query_threshold = 1.0  # seconds
        self.analysis_batch_size = 1000
    
    async def analyze_memory_patterns(self, session_id: Optional[str] = None,
                                    time_window_hours: int = 24) -> MemoryAnalytics:
        """Analyze memory storage patterns and usage."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            if db_manager.is_postgresql():
                analytics = await self._analyze_postgresql_memory_patterns(session_id, cutoff_time)
            else:
                analytics = await self._analyze_sqlite_memory_patterns(session_id, cutoff_time)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to analyze memory patterns: {e}")
            return MemoryAnalytics(
                total_memories=0,
                memories_by_entity={},
                memories_by_session={},
                memories_by_protocol={},
                emotional_salience_distribution={},
                memory_type_distribution={},
                average_memory_length=0.0,
                recent_activity=[]
            )
    
    async def _analyze_postgresql_memory_patterns(self, session_id: Optional[str], 
                                                cutoff_time: datetime) -> MemoryAnalytics:
        """Analyze memory patterns in PostgreSQL."""
        from mcp_server.database.postgres_database import postgres_db
        
        # Base query conditions
        conditions = ["m.created_at >= $1"]
        params = [cutoff_time]
        param_count = 1
        
        if session_id:
            param_count += 1
            conditions.append(f"m.session_id = ${param_count}")
            params.append(session_id)
        
        where_clause = " AND ".join(conditions)
        
        # Total memories
        total_query = f"SELECT COUNT(*) FROM memories m WHERE {where_clause}"
        total_result = await postgres_db.execute_query(total_query, *params)
        total_memories = total_result[0]['count'] if total_result else 0
        
        # Memories by entity
        entity_query = f"""
            SELECT e.name, COUNT(*) as count
            FROM memories m
            JOIN entities e ON m.entity_id = e.id
            WHERE {where_clause}
            GROUP BY e.name
            ORDER BY count DESC
        """
        entity_results = await postgres_db.execute_query(entity_query, *params)
        memories_by_entity = {row['name']: row['count'] for row in entity_results}
        
        # Memories by session
        session_query = f"""
            SELECT m.session_id, COUNT(*) as count
            FROM memories m
            WHERE {where_clause}
            GROUP BY m.session_id
            ORDER BY count DESC
        """
        session_results = await postgres_db.execute_query(session_query, *params)
        memories_by_session = {row['session_id']: row['count'] for row in session_results}
        
        # Memories by protocol
        protocol_query = f"""
            SELECT m.source_protocol, COUNT(*) as count
            FROM memories m
            WHERE {where_clause}
            GROUP BY m.source_protocol
            ORDER BY count DESC
        """
        protocol_results = await postgres_db.execute_query(protocol_query, *params)
        memories_by_protocol = {row['source_protocol']: row['count'] for row in protocol_results}
        
        # Emotional salience distribution
        salience_query = f"""
            SELECT 
                CASE 
                    WHEN m.emotional_salience < 0.3 THEN 'low'
                    WHEN m.emotional_salience < 0.7 THEN 'medium'
                    ELSE 'high'
                END as salience_level,
                COUNT(*) as count
            FROM memories m
            WHERE {where_clause}
            GROUP BY salience_level
        """
        salience_results = await postgres_db.execute_query(salience_query, *params)
        emotional_salience_distribution = {row['salience_level']: row['count'] for row in salience_results}
        
        # Memory type distribution
        type_query = f"""
            SELECT m.memory_type, COUNT(*) as count
            FROM memories m
            WHERE {where_clause}
            GROUP BY m.memory_type
            ORDER BY count DESC
        """
        type_results = await postgres_db.execute_query(type_query, *params)
        memory_type_distribution = {row['memory_type']: row['count'] for row in type_results}
        
        # Average memory length
        length_query = f"""
            SELECT AVG(LENGTH(m.content)) as avg_length
            FROM memories m
            WHERE {where_clause}
        """
        length_result = await postgres_db.execute_query(length_query, *params)
        average_memory_length = float(length_result[0]['avg_length'] or 0)
        
        # Recent activity
        activity_query = f"""
            SELECT m.id, m.content, e.name as entity_name, m.created_at,
                   m.emotional_salience, m.memory_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.id
            WHERE {where_clause}
            ORDER BY m.created_at DESC
            LIMIT 20
        """
        activity_results = await postgres_db.execute_query(activity_query, *params)
        recent_activity = [
            {
                'memory_id': row['id'],
                'content': row['content'][:100] + '...' if len(row['content']) > 100 else row['content'],
                'entity_name': row['entity_name'],
                'created_at': row['created_at'].isoformat() if hasattr(row['created_at'], 'isoformat') else str(row['created_at']),
                'emotional_salience': row['emotional_salience'],
                'memory_type': row['memory_type']
            }
            for row in activity_results
        ]
        
        return MemoryAnalytics(
            total_memories=total_memories,
            memories_by_entity=memories_by_entity,
            memories_by_session=memories_by_session,
            memories_by_protocol=memories_by_protocol,
            emotional_salience_distribution=emotional_salience_distribution,
            memory_type_distribution=memory_type_distribution,
            average_memory_length=average_memory_length,
            recent_activity=recent_activity
        )
    
    async def _analyze_sqlite_memory_patterns(self, session_id: Optional[str], 
                                            cutoff_time: datetime) -> MemoryAnalytics:
        """Analyze memory patterns in SQLite."""
        conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
        if not conn:
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
        
        if not conn:
            raise RuntimeError("Could not establish SQLite connection")
        
        try:
            cursor = conn.cursor()
            
            # Base query conditions
            conditions = ["m.timestamp >= ?"]
            params = [cutoff_time.isoformat()]
            
            if session_id:
                conditions.append("m.session_id = ?")
                params.append(session_id)
            
            where_clause = " AND ".join(conditions)
            
            # Total memories
            cursor.execute(f"SELECT COUNT(*) FROM memories m WHERE {where_clause}", params)
            total_memories = cursor.fetchone()[0]
            
            # Memories by entity
            cursor.execute(f"""
                SELECT e.name, COUNT(*) as count
                FROM memories m
                JOIN entities e ON m.entity_id = e.id
                WHERE {where_clause}
                GROUP BY e.name
                ORDER BY count DESC
            """, params)
            memories_by_entity = dict(cursor.fetchall())
            
            # Memories by session
            cursor.execute(f"""
                SELECT m.session_id, COUNT(*) as count
                FROM memories m
                WHERE {where_clause}
                GROUP BY m.session_id
                ORDER BY count DESC
            """, params)
            memories_by_session = dict(cursor.fetchall())
            
            # Memories by protocol
            cursor.execute(f"""
                SELECT m.source_protocol, COUNT(*) as count
                FROM memories m
                WHERE {where_clause}
                GROUP BY m.source_protocol
                ORDER BY count DESC
            """, params)
            memories_by_protocol = dict(cursor.fetchall())
            
            # Emotional salience distribution
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN m.emotional_salience < 0.3 THEN 'low'
                        WHEN m.emotional_salience < 0.7 THEN 'medium'
                        ELSE 'high'
                    END as salience_level,
                    COUNT(*) as count
                FROM memories m
                WHERE {where_clause}
                GROUP BY salience_level
            """, params)
            emotional_salience_distribution = dict(cursor.fetchall())
            
            # Memory type distribution
            cursor.execute(f"""
                SELECT m.memory_type, COUNT(*) as count
                FROM memories m
                WHERE {where_clause}
                GROUP BY m.memory_type
                ORDER BY count DESC
            """, params)
            memory_type_distribution = dict(cursor.fetchall())
            
            # Average memory length
            cursor.execute(f"""
                SELECT AVG(LENGTH(m.content)) as avg_length
                FROM memories m
                WHERE {where_clause}
            """, params)
            result = cursor.fetchone()
            average_memory_length = float(result[0] or 0)
            
            # Recent activity
            cursor.execute(f"""
                SELECT m.id, m.content, e.name as entity_name, m.timestamp,
                       m.emotional_salience, m.memory_type
                FROM memories m
                JOIN entities e ON m.entity_id = e.id
                WHERE {where_clause}
                ORDER BY m.timestamp DESC
                LIMIT 20
            """, params)
            
            recent_activity = []
            for row in cursor.fetchall():
                recent_activity.append({
                    'memory_id': row[0],
                    'content': row[1][:100] + '...' if len(row[1]) > 100 else row[1],
                    'entity_name': row[2],
                    'created_at': row[3],
                    'emotional_salience': row[4],
                    'memory_type': row[5]
                })
            
            return MemoryAnalytics(
                total_memories=total_memories,
                memories_by_entity=memories_by_entity,
                memories_by_session=memories_by_session,
                memories_by_protocol=memories_by_protocol,
                emotional_salience_distribution=emotional_salience_distribution,
                memory_type_distribution=memory_type_distribution,
                average_memory_length=average_memory_length,
                recent_activity=recent_activity
            )
            
        finally:
            conn.close()
    
    async def analyze_query_patterns(self, time_window_hours: int = 24) -> List[QueryPattern]:
        """Analyze database query patterns and performance."""
        try:
            # This would integrate with the performance monitor
            from mcp_server.database.performance_monitor import performance_monitor
            
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Get query history from performance monitor
            query_metrics = [m for m in performance_monitor.query_history if m.timestamp >= cutoff_time]
            
            if not query_metrics:
                return []
            
            # Group queries by hash (similar queries)
            query_groups = defaultdict(list)
            for metric in query_metrics:
                query_groups[metric.query_hash].append(metric)
            
            patterns = []
            for query_hash, metrics in query_groups.items():
                if not metrics:
                    continue
                
                # Calculate statistics
                execution_times = [m.execution_time for m in metrics]
                error_count = sum(1 for m in metrics if not m.success)
                
                pattern = QueryPattern(
                    pattern_hash=query_hash,
                    normalized_query="<normalized query>",  # Would need actual query normalization
                    query_type=metrics[0].query_type,
                    table_names=metrics[0].table_names,
                    execution_count=len(metrics),
                    avg_duration=statistics.mean(execution_times),
                    min_duration=min(execution_times),
                    max_duration=max(execution_times),
                    total_duration=sum(execution_times),
                    error_count=error_count,
                    last_execution=max(m.timestamp for m in metrics),
                    first_seen=min(m.timestamp for m in metrics)
                )
                
                patterns.append(pattern)
            
            # Sort by total duration (most resource-intensive first)
            patterns.sort(key=lambda p: p.total_duration, reverse=True)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to analyze query patterns: {e}")
            return []
    
    async def get_performance_metrics(self, time_window_hours: int = 1) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        try:
            from mcp_server.database.performance_monitor import performance_monitor
            
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Get recent query metrics
            recent_metrics = [m for m in performance_monitor.query_history if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return PerformanceMetrics(
                    queries_per_second=0.0,
                    average_response_time=0.0,
                    p95_response_time=0.0,
                    p99_response_time=0.0,
                    error_rate=0.0,
                    throughput_trend=[],
                    slow_query_count=0,
                    connection_utilization=0.0
                )
            
            # Calculate metrics
            total_queries = len(recent_metrics)
            successful_queries = sum(1 for m in recent_metrics if m.success)
            
            time_span_seconds = time_window_hours * 3600
            queries_per_second = total_queries / time_span_seconds
            
            execution_times = [m.execution_time for m in recent_metrics]
            average_response_time = statistics.mean(execution_times)
            
            # Calculate percentiles
            sorted_times = sorted(execution_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else sorted_times[-1]
            
            error_rate = (total_queries - successful_queries) / total_queries if total_queries > 0 else 0.0
            
            slow_query_count = sum(1 for t in execution_times if t > self.slow_query_threshold)
            
            # Calculate throughput trend (queries per minute over time)
            throughput_trend = []
            start_time = min(m.timestamp for m in recent_metrics)
            
            # Group by 5-minute intervals
            interval_minutes = 5
            current_time = start_time
            end_time = datetime.now()
            
            while current_time < end_time:
                interval_end = current_time + timedelta(minutes=interval_minutes)
                interval_queries = [m for m in recent_metrics 
                                  if current_time <= m.timestamp < interval_end]
                
                queries_per_minute = len(interval_queries) / interval_minutes
                throughput_trend.append((current_time, queries_per_minute))
                
                current_time = interval_end
            
            # Get connection utilization (if available)
            connection_utilization = 0.0
            try:
                from mcp_server.database.connection_monitor import connection_monitor
                current_metrics = connection_monitor.get_current_metrics()
                if current_metrics:
                    connection_utilization = current_metrics.connection_utilization
            except Exception:
                pass
            
            return PerformanceMetrics(
                queries_per_second=queries_per_second,
                average_response_time=average_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                error_rate=error_rate,
                throughput_trend=throughput_trend,
                slow_query_count=slow_query_count,
                connection_utilization=connection_utilization
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(
                queries_per_second=0.0,
                average_response_time=0.0,
                p95_response_time=0.0,
                p99_response_time=0.0,
                error_rate=0.0,
                throughput_trend=[],
                slow_query_count=0,
                connection_utilization=0.0
            )
    
    async def generate_analytics_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        try:
            # Collect all analytics data
            memory_analytics = await self.analyze_memory_patterns(session_id)
            query_patterns = await self.analyze_query_patterns()
            performance_metrics = await self.get_performance_metrics()
            
            # Get database health information
            db_health = await db_manager.health_check()
            
            # Calculate trends and insights
            insights = await self._generate_insights(memory_analytics, query_patterns, performance_metrics)
            
            report = {
                "report_generated_at": datetime.now().isoformat(),
                "session_filter": session_id,
                "database_type": db_manager.get_database_type().value,
                "database_health": db_health,
                "memory_analytics": {
                    "total_memories": memory_analytics.total_memories,
                    "memories_by_entity": memory_analytics.memories_by_entity,
                    "memories_by_session": memory_analytics.memories_by_session,
                    "memories_by_protocol": memory_analytics.memories_by_protocol,
                    "emotional_salience_distribution": memory_analytics.emotional_salience_distribution,
                    "memory_type_distribution": memory_analytics.memory_type_distribution,
                    "average_memory_length": memory_analytics.average_memory_length,
                    "recent_activity_count": len(memory_analytics.recent_activity)
                },
                "query_performance": {
                    "total_query_patterns": len(query_patterns),
                    "top_resource_intensive": [
                        {
                            "pattern_hash": p.pattern_hash,
                            "query_type": p.query_type,
                            "execution_count": p.execution_count,
                            "total_duration": p.total_duration,
                            "avg_duration": p.avg_duration,
                            "error_count": p.error_count
                        }
                        for p in query_patterns[:10]
                    ]
                },
                "performance_metrics": {
                    "queries_per_second": performance_metrics.queries_per_second,
                    "average_response_time": performance_metrics.average_response_time,
                    "p95_response_time": performance_metrics.p95_response_time,
                    "p99_response_time": performance_metrics.p99_response_time,
                    "error_rate": performance_metrics.error_rate,
                    "slow_query_count": performance_metrics.slow_query_count,
                    "connection_utilization": performance_metrics.connection_utilization
                },
                "insights_and_recommendations": insights
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate analytics report: {e}")
            return {
                "error": str(e),
                "report_generated_at": datetime.now().isoformat()
            }
    
    async def _generate_insights(self, memory_analytics: MemoryAnalytics, 
                               query_patterns: List[QueryPattern],
                               performance_metrics: PerformanceMetrics) -> List[Dict[str, str]]:
        """Generate insights and recommendations based on analytics."""
        insights = []
        
        # Memory insights
        if memory_analytics.total_memories > 10000:
            insights.append({
                "category": "memory_volume",
                "level": "info",
                "insight": f"Large memory dataset ({memory_analytics.total_memories:,} memories)",
                "recommendation": "Consider implementing memory archiving or partitioning strategies"
            })
        
        # Entity distribution insights
        if memory_analytics.memories_by_entity:
            top_entity_count = max(memory_analytics.memories_by_entity.values())
            total_memories = memory_analytics.total_memories
            
            if top_entity_count > total_memories * 0.5:
                top_entity = max(memory_analytics.memories_by_entity.items(), key=lambda x: x[1])
                insights.append({
                    "category": "memory_distribution",
                    "level": "warning",
                    "insight": f"Memory distribution is heavily skewed towards '{top_entity[0]}' ({top_entity[1]} memories)",
                    "recommendation": "Consider balancing memory distribution or reviewing entity classification"
                })
        
        # Performance insights
        if performance_metrics.error_rate > 0.05:  # 5% error rate
            insights.append({
                "category": "reliability",
                "level": "warning",
                "insight": f"High error rate detected: {performance_metrics.error_rate:.1%}",
                "recommendation": "Review failed queries and improve error handling"
            })
        
        if performance_metrics.average_response_time > 1.0:
            insights.append({
                "category": "performance",
                "level": "warning",
                "insight": f"Average response time is high: {performance_metrics.average_response_time:.2f}s",
                "recommendation": "Consider query optimization or adding database indexes"
            })
        
        if performance_metrics.connection_utilization > 0.8:
            insights.append({
                "category": "resources",
                "level": "warning",
                "insight": f"Connection utilization is high: {performance_metrics.connection_utilization:.1%}",
                "recommendation": "Consider increasing connection pool size or optimizing query patterns"
            })
        
        # Query pattern insights
        if query_patterns:
            most_frequent = max(query_patterns, key=lambda p: p.execution_count)
            if most_frequent.execution_count > 1000:  # Very frequent query
                insights.append({
                    "category": "query_optimization",
                    "level": "info",
                    "insight": f"Very frequent query pattern detected ({most_frequent.execution_count} executions)",
                    "recommendation": "Consider caching results for this query pattern"
                })
        
        # Emotional salience insights
        if memory_analytics.emotional_salience_distribution:
            high_salience = memory_analytics.emotional_salience_distribution.get('high', 0)
            total = sum(memory_analytics.emotional_salience_distribution.values())
            
            if total > 0 and high_salience / total < 0.1:  # Less than 10% high salience
                insights.append({
                    "category": "memory_quality",
                    "level": "info",
                    "insight": f"Low proportion of high-salience memories: {high_salience/total:.1%}",
                    "recommendation": "Review emotional salience calculation or memory filtering criteria"
                })
        
        return insights
    
    async def get_memory_growth_trend(self, days: int = 30) -> Dict[str, Any]:
        """Analyze memory growth trends over time."""
        try:
            # Get daily memory counts for the specified period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                query = """
                    SELECT DATE(created_at) as date, COUNT(*) as count
                    FROM memories
                    WHERE created_at >= $1 AND created_at <= $2
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """
                
                rows = await postgres_db.execute_query(query, start_date, end_date)
                
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return {"error": "Could not establish database connection"}
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT DATE(timestamp) as date, COUNT(*) as count
                        FROM memories
                        WHERE timestamp >= ? AND timestamp <= ?
                        GROUP BY DATE(timestamp)
                        ORDER BY date
                    """, (start_date.isoformat(), end_date.isoformat()))
                    
                    columns = [desc[0] for desc in cursor.description]
                    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    
                finally:
                    conn.close()
            
            # Process data
            daily_counts = [(row['date'], row['count']) for row in rows]
            
            # Calculate trend
            if len(daily_counts) > 1:
                counts = [count for _, count in daily_counts]
                trend = "increasing" if counts[-1] > counts[0] else "decreasing" if counts[-1] < counts[0] else "stable"
                avg_daily_growth = sum(counts) / len(counts)
            else:
                trend = "insufficient_data"
                avg_daily_growth = 0
            
            return {
                "analysis_period_days": days,
                "daily_memory_counts": daily_counts,
                "trend": trend,
                "average_daily_memories": avg_daily_growth,
                "total_new_memories": sum(count for _, count in daily_counts)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze memory growth trend: {e}")
            return {"error": str(e)}
    
    async def analyze_session_patterns(self) -> Dict[str, Any]:
        """Analyze session-based memory patterns."""
        try:
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                
                # Session statistics
                session_stats_query = """
                    SELECT 
                        session_id,
                        COUNT(*) as memory_count,
                        AVG(emotional_salience) as avg_salience,
                        MIN(created_at) as first_memory,
                        MAX(created_at) as last_memory,
                        COUNT(DISTINCT entity_id) as unique_entities
                    FROM memories
                    WHERE session_id IS NOT NULL
                    GROUP BY session_id
                    ORDER BY memory_count DESC
                """
                
                session_stats = await postgres_db.execute_query(session_stats_query)
                
            else:
                conn = db_manager.get_db_connection() if hasattr(db_manager, 'get_db_connection') else None
                if not conn:
                    from mcp_server.database.memory_database import get_db_connection
                    conn = get_db_connection()
                
                if not conn:
                    return {"error": "Could not establish database connection"}
                
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT 
                            session_id,
                            COUNT(*) as memory_count,
                            AVG(emotional_salience) as avg_salience,
                            MIN(timestamp) as first_memory,
                            MAX(timestamp) as last_memory,
                            COUNT(DISTINCT entity_id) as unique_entities
                        FROM memories
                        WHERE session_id IS NOT NULL
                        GROUP BY session_id
                        ORDER BY memory_count DESC
                    """)
                    
                    columns = [desc[0] for desc in cursor.description]
                    session_stats = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    
                finally:
                    conn.close()
            
            # Process session data
            total_sessions = len(session_stats)
            
            if total_sessions == 0:
                return {"message": "No session data available"}
            
            memory_counts = [s['memory_count'] for s in session_stats]
            avg_memories_per_session = statistics.mean(memory_counts)
            
            # Find most active session
            most_active_session = max(session_stats, key=lambda s: s['memory_count'])
            
            # Calculate session duration statistics
            session_durations = []
            for session in session_stats:
                if session['first_memory'] and session['last_memory']:
                    first = datetime.fromisoformat(str(session['first_memory']).replace('Z', '+00:00'))
                    last = datetime.fromisoformat(str(session['last_memory']).replace('Z', '+00:00'))
                    duration = (last - first).total_seconds() / 3600  # hours
                    session_durations.append(duration)
            
            avg_session_duration = statistics.mean(session_durations) if session_durations else 0
            
            return {
                "total_sessions": total_sessions,
                "average_memories_per_session": avg_memories_per_session,
                "most_active_session": {
                    "session_id": most_active_session['session_id'],
                    "memory_count": most_active_session['memory_count'],
                    "avg_salience": most_active_session['avg_salience'],
                    "unique_entities": most_active_session['unique_entities']
                },
                "average_session_duration_hours": avg_session_duration,
                "session_memory_distribution": {
                    "min": min(memory_counts),
                    "max": max(memory_counts),
                    "median": statistics.median(memory_counts)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze session patterns: {e}")
            return {"error": str(e)}

# Global analytics engine instance
analytics_engine = DatabaseAnalyticsEngine()

async def get_analytics_engine() -> DatabaseAnalyticsEngine:
    """Get the analytics engine instance."""
    return analytics_engine