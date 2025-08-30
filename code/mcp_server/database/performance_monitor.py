import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import json

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

@dataclass
class QueryMetrics:
    """Metrics for a database query execution."""
    query_hash: str
    execution_time: float
    timestamp: datetime
    row_count: int
    query_type: str  # SELECT, INSERT, UPDATE, DELETE
    table_names: List[str]
    success: bool
    error_message: Optional[str] = None

class DatabasePerformanceMonitor:
    """
    Advanced database performance monitoring and optimization system.
    Tracks query performance, identifies slow queries, and provides optimization recommendations.
    """
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.query_history: deque = deque(maxlen=max_history_size)
        self.query_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_executions': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0,
            'last_execution': None,
            'recent_times': deque(maxlen=100)
        })
        self.slow_query_threshold = 1.0  # seconds
        self.monitoring_enabled = True
    
    def _extract_query_info(self, query: str) -> tuple:
        """Extract basic information from SQL query."""
        query_upper = query.upper().strip()
        
        # Determine query type
        query_type = 'UNKNOWN'
        for qtype in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']:
            if query_upper.startswith(qtype):
                query_type = qtype
                break
        
        # Extract table names (basic implementation)
        table_names = []
        words = query_upper.split()
        for i, word in enumerate(words):
            if word in ['FROM', 'INTO', 'UPDATE', 'JOIN']:
                if i + 1 < len(words):
                    table_name = words[i + 1].split('(')[0].strip()
                    if table_name not in ['SELECT', 'WHERE', '(']:
                        table_names.append(table_name.lower())
        
        # Create query hash for grouping similar queries
        normalized_query = self._normalize_query(query)
        query_hash = str(hash(normalized_query))
        
        return query_hash, query_type, table_names
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query by removing specific values to group similar queries."""
        import re
        
        # Remove string literals
        normalized = re.sub(r"'[^']*'", "'?'", query)
        
        # Remove numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        
        # Remove multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip().upper()
    
    async def monitor_query(self, query_func: Callable, query: str, *args, **kwargs):
        """Monitor the execution of a database query."""
        if not self.monitoring_enabled:
            return await query_func(query, *args, **kwargs)
        
        start_time = time.time()
        query_hash, query_type, table_names = self._extract_query_info(query)
        
        try:
            result = await query_func(query, *args, **kwargs)
            execution_time = time.time() - start_time
            
            # Determine row count
            row_count = 0
            if isinstance(result, list):
                row_count = len(result)
            elif hasattr(result, '__len__'):
                row_count = len(result)
            
            # Create metrics
            metrics = QueryMetrics(
                query_hash=query_hash,
                execution_time=execution_time,
                timestamp=datetime.now(),
                row_count=row_count,
                query_type=query_type,
                table_names=table_names,
                success=True
            )
            
            # Update statistics
            self._update_stats(metrics)
            
            # Log slow queries
            if execution_time > self.slow_query_threshold:
                logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}...")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            metrics = QueryMetrics(
                query_hash=query_hash,
                execution_time=execution_time,
                timestamp=datetime.now(),
                row_count=0,
                query_type=query_type,
                table_names=table_names,
                success=False,
                error_message=str(e)
            )
            
            self._update_stats(metrics)
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _update_stats(self, metrics: QueryMetrics):
        """Update query statistics."""
        self.query_history.append(metrics)
        
        stats = self.query_stats[metrics.query_hash]
        stats['total_executions'] += 1
        stats['total_time'] += metrics.execution_time
        stats['avg_time'] = stats['total_time'] / stats['total_executions']
        stats['min_time'] = min(stats['min_time'], metrics.execution_time)
        stats['max_time'] = max(stats['max_time'], metrics.execution_time)
        stats['last_execution'] = metrics.timestamp
        stats['recent_times'].append(metrics.execution_time)
        
        if not metrics.success:
            stats['error_count'] += 1
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_metrics = [m for m in self.query_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No queries executed in the specified time window"}
        
        # Calculate overall statistics
        total_queries = len(recent_metrics)
        successful_queries = sum(1 for m in recent_metrics if m.success)
        failed_queries = total_queries - successful_queries
        
        execution_times = [m.execution_time for m in recent_metrics]
        avg_execution_time = statistics.mean(execution_times)
        median_execution_time = statistics.median(execution_times)
        p95_execution_time = statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times)
        
        # Query type breakdown
        query_type_counts = defaultdict(int)
        for m in recent_metrics:
            query_type_counts[m.query_type] += 1
        
        # Table access patterns
        table_access_counts = defaultdict(int)
        for m in recent_metrics:
            for table in m.table_names:
                table_access_counts[table] += 1
        
        # Slow query analysis
        slow_queries = [m for m in recent_metrics if m.execution_time > self.slow_query_threshold]
        
        return {
            "time_window_hours": time_window_hours,
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": (successful_queries / total_queries) * 100 if total_queries > 0 else 0,
            "performance_metrics": {
                "avg_execution_time": avg_execution_time,
                "median_execution_time": median_execution_time,
                "p95_execution_time": p95_execution_time,
                "total_execution_time": sum(execution_times)
            },
            "query_type_distribution": dict(query_type_counts),
            "table_access_patterns": dict(sorted(table_access_counts.items(), key=lambda x: x[1], reverse=True)),
            "slow_queries": {
                "count": len(slow_queries),
                "threshold_seconds": self.slow_query_threshold,
                "slowest_query_time": max(slow_queries, key=lambda x: x.execution_time).execution_time if slow_queries else 0
            }
        }
    
    def get_slow_queries_report(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get detailed report of slowest queries."""
        # Group by query hash and find slowest instances
        query_groups = defaultdict(list)
        for metrics in self.query_history:
            query_groups[metrics.query_hash].append(metrics)
        
        slow_query_reports = []
        for query_hash, metrics_list in query_groups.items():
            if not metrics_list:
                continue
            
            stats = self.query_stats[query_hash]
            slowest_instance = max(metrics_list, key=lambda x: x.execution_time)
            
            if stats['avg_time'] > self.slow_query_threshold or slowest_instance.execution_time > self.slow_query_threshold:
                slow_query_reports.append({
                    "query_hash": query_hash,
                    "avg_execution_time": stats['avg_time'],
                    "max_execution_time": stats['max_time'],
                    "min_execution_time": stats['min_time'],
                    "total_executions": stats['total_executions'],
                    "error_count": stats['error_count'],
                    "last_execution": stats['last_execution'].isoformat() if stats['last_execution'] else None,
                    "query_type": slowest_instance.query_type,
                    "affected_tables": slowest_instance.table_names,
                    "recent_avg_time": statistics.mean(stats['recent_times']) if stats['recent_times'] else 0
                })
        
        # Sort by average execution time
        slow_query_reports.sort(key=lambda x: x['avg_execution_time'], reverse=True)
        return slow_query_reports[:limit]
    
    def get_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate database optimization recommendations based on collected metrics."""
        recommendations = []
        
        # Analyze slow queries
        slow_queries = self.get_slow_queries_report()
        
        for query in slow_queries[:5]:  # Top 5 slow queries
            if query['avg_execution_time'] > 2.0:
                recommendations.append({
                    "type": "query_optimization",
                    "severity": "high",
                    "description": f"Query with hash {query['query_hash']} has high average execution time ({query['avg_execution_time']:.3f}s)",
                    "suggestion": "Consider adding appropriate indexes on frequently queried columns or optimizing the query structure",
                    "affected_tables": ", ".join(query['affected_tables'])
                })
        
        # Analyze table access patterns
        summary = self.get_performance_summary()
        table_access = summary.get('table_access_patterns', {})
        
        # Recommend indexes for frequently accessed tables
        for table, access_count in list(table_access.items())[:3]:
            if access_count > 100:  # Frequently accessed
                recommendations.append({
                    "type": "indexing",
                    "severity": "medium",
                    "description": f"Table '{table}' is accessed frequently ({access_count} times)",
                    "suggestion": "Consider adding indexes on commonly filtered columns for this table",
                    "affected_tables": table
                })
        
        # Check for high error rates
        if summary.get('success_rate', 100) < 95:
            recommendations.append({
                "type": "error_handling",
                "severity": "high",
                "description": f"Query success rate is low ({summary['success_rate']:.1f}%)",
                "suggestion": "Review failed queries and improve error handling or query validation",
                "affected_tables": "all"
            })
        
        # Check connection pool performance
        if db_manager.is_postgresql():
            recommendations.append({
                "type": "connection_pooling",
                "severity": "medium",
                "description": "Monitor connection pool utilization for PostgreSQL",
                "suggestion": "Consider adjusting min_size and max_size parameters based on usage patterns",
                "affected_tables": "all"
            })
        
        return recommendations
    
    async def analyze_table_statistics(self) -> Dict[str, Any]:
        """Analyze table-level statistics and performance."""
        if not db_manager.is_postgresql():
            return {"message": "Table statistics analysis only available for PostgreSQL"}
        
        try:
            # Get table size information
            table_stats_query = """
                SELECT 
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats 
                WHERE schemaname = 'public'
                ORDER BY tablename, attname;
            """
            
            table_sizes_query = """
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                    pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
            """
            
            from mcp_server.database.postgres_database import postgres_db
            
            table_stats = await postgres_db.execute_query(table_stats_query)
            table_sizes = await postgres_db.execute_query(table_sizes_query)
            
            return {
                "table_statistics": table_stats,
                "table_sizes": table_sizes,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze table statistics: {e}")
            return {"error": str(e)}
    
    def set_slow_query_threshold(self, threshold_seconds: float):
        """Set the threshold for identifying slow queries."""
        self.slow_query_threshold = threshold_seconds
        logger.info(f"Slow query threshold set to {threshold_seconds} seconds")
    
    def enable_monitoring(self):
        """Enable performance monitoring."""
        self.monitoring_enabled = True
        logger.info("Database performance monitoring enabled")
    
    def disable_monitoring(self):
        """Disable performance monitoring."""
        self.monitoring_enabled = False
        logger.info("Database performance monitoring disabled")
    
    def clear_history(self):
        """Clear performance history."""
        self.query_history.clear()
        self.query_stats.clear()
        logger.info("Performance monitoring history cleared")

# Global performance monitor instance
performance_monitor = DatabasePerformanceMonitor()

async def get_performance_monitor() -> DatabasePerformanceMonitor:
    """Get the performance monitor instance."""
    return performance_monitor