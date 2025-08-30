import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os

from mcp_server.database.database_manager import db_manager

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNAVAILABLE = "unavailable"

@dataclass
class ConnectionMetrics:
    """Metrics for database connection monitoring."""
    timestamp: datetime
    active_connections: int
    idle_connections: int
    total_connections: int
    connection_utilization: float
    avg_query_time: float
    failed_connections: int
    connection_errors: List[str]
    pool_status: ConnectionStatus

@dataclass
class PoolConfiguration:
    """Configuration parameters for connection pool optimization."""
    min_connections: int
    max_connections: int
    connection_timeout: int
    idle_timeout: int
    max_lifetime: int
    health_check_interval: int
    retry_attempts: int
    retry_delay: int

class DatabaseConnectionMonitor:
    """
    Advanced database connection pool monitoring and optimization system.
    Monitors connection health, optimizes pool configuration, and provides alerting.
    """
    
    def __init__(self):
        self.metrics_history: List[ConnectionMetrics] = []
        self.max_history_size = 1440  # 24 hours of minute-by-minute data
        self.monitoring_enabled = True
        self.health_check_interval = 60  # seconds
        self.alert_thresholds = {
            'utilization_warning': 0.8,
            'utilization_critical': 0.95,
            'avg_query_time_warning': 1.0,  # seconds
            'avg_query_time_critical': 5.0,  # seconds
            'error_rate_warning': 0.05,  # 5%
            'error_rate_critical': 0.10   # 10%
        }
        self.connection_errors = []
        self.last_health_check = None
        self._monitor_task = None
    
    async def start_monitoring(self):
        """Start continuous connection monitoring."""
        if self._monitor_task and not self._monitor_task.done():
            logger.warning("Connection monitoring already running")
            return
        
        self.monitoring_enabled = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Database connection monitoring started")
    
    async def stop_monitoring(self):
        """Stop connection monitoring."""
        self.monitoring_enabled = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Database connection monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_enabled:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection monitoring loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def collect_metrics(self) -> ConnectionMetrics:
        """Collect current connection metrics."""
        try:
            current_time = datetime.now()
            
            if db_manager.is_postgresql():
                metrics = await self._collect_postgresql_metrics()
            else:
                metrics = await self._collect_sqlite_metrics()
            
            # Add to history
            self.metrics_history.append(metrics)
            
            # Trim history if too large
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            self.last_health_check = current_time
            
            # Check for alerts
            await self._check_alerts(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
            
            # Return degraded metrics
            return ConnectionMetrics(
                timestamp=datetime.now(),
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                connection_utilization=0.0,
                avg_query_time=0.0,
                failed_connections=1,
                connection_errors=[str(e)],
                pool_status=ConnectionStatus.UNAVAILABLE
            )
    
    async def _collect_postgresql_metrics(self) -> ConnectionMetrics:
        """Collect PostgreSQL connection pool metrics."""
        from mcp_server.database.postgres_database import postgres_db
        
        try:
            # Get pool statistics
            pool = postgres_db.pool
            if not pool:
                return ConnectionMetrics(
                    timestamp=datetime.now(),
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    connection_utilization=0.0,
                    avg_query_time=0.0,
                    failed_connections=0,
                    connection_errors=[],
                    pool_status=ConnectionStatus.UNAVAILABLE
                )
            
            # Pool metrics
            total_connections = pool.get_size()
            free_connections = pool.get_free_size()
            active_connections = total_connections - free_connections
            max_connections = pool.get_max_size()
            
            connection_utilization = (total_connections / max_connections) if max_connections > 0 else 0.0
            
            # Database-level statistics
            async with pool.acquire() as conn:
                db_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_connections,
                        COUNT(*) FILTER (WHERE state = 'active') as active_queries,
                        AVG(EXTRACT(EPOCH FROM (now() - query_start))) as avg_query_duration
                    FROM pg_stat_activity
                    WHERE datname = current_database()
                    AND state IS NOT NULL
                """)
                
                # Connection statistics
                conn_stats = await conn.fetchrow("""
                    SELECT 
                        sum(numbackends) as backends,
                        sum(xact_commit + xact_rollback) as total_transactions,
                        sum(xact_rollback) as rollbacks
                    FROM pg_stat_database 
                    WHERE datname = current_database()
                """)
            
            avg_query_time = float(db_stats['avg_query_duration'] or 0.0)
            
            # Determine status
            status = self._determine_connection_status(
                connection_utilization, avg_query_time, 0
            )
            
            return ConnectionMetrics(
                timestamp=datetime.now(),
                active_connections=active_connections,
                idle_connections=free_connections,
                total_connections=total_connections,
                connection_utilization=connection_utilization,
                avg_query_time=avg_query_time,
                failed_connections=0,
                connection_errors=[],
                pool_status=status
            )
            
        except Exception as e:
            logger.error(f"Failed to collect PostgreSQL metrics: {e}")
            raise
    
    async def _collect_sqlite_metrics(self) -> ConnectionMetrics:
        """Collect SQLite connection metrics (simplified)."""
        try:
            # For SQLite, we don't have a real connection pool
            # But we can simulate metrics based on connection attempts
            
            start_time = time.time()
            
            # Test connection
            from mcp_server.database.memory_database import get_db_connection
            conn = get_db_connection()
            
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                    query_time = time.time() - start_time
                    
                    status = ConnectionStatus.HEALTHY if query_time < 1.0 else ConnectionStatus.DEGRADED
                    
                    return ConnectionMetrics(
                        timestamp=datetime.now(),
                        active_connections=1,
                        idle_connections=0,
                        total_connections=1,
                        connection_utilization=1.0,
                        avg_query_time=query_time,
                        failed_connections=0,
                        connection_errors=[],
                        pool_status=status
                    )
                finally:
                    conn.close()
            else:
                return ConnectionMetrics(
                    timestamp=datetime.now(),
                    active_connections=0,
                    idle_connections=0,
                    total_connections=0,
                    connection_utilization=0.0,
                    avg_query_time=0.0,
                    failed_connections=1,
                    connection_errors=["Failed to establish SQLite connection"],
                    pool_status=ConnectionStatus.UNAVAILABLE
                )
                
        except Exception as e:
            return ConnectionMetrics(
                timestamp=datetime.now(),
                active_connections=0,
                idle_connections=0,
                total_connections=0,
                connection_utilization=0.0,
                avg_query_time=0.0,
                failed_connections=1,
                connection_errors=[str(e)],
                pool_status=ConnectionStatus.UNAVAILABLE
            )
    
    def _determine_connection_status(self, utilization: float, avg_query_time: float, 
                                   error_rate: float) -> ConnectionStatus:
        """Determine overall connection status based on metrics."""
        if (utilization >= self.alert_thresholds['utilization_critical'] or 
            avg_query_time >= self.alert_thresholds['avg_query_time_critical'] or
            error_rate >= self.alert_thresholds['error_rate_critical']):
            return ConnectionStatus.CRITICAL
        
        elif (utilization >= self.alert_thresholds['utilization_warning'] or
              avg_query_time >= self.alert_thresholds['avg_query_time_warning'] or
              error_rate >= self.alert_thresholds['error_rate_warning']):
            return ConnectionStatus.DEGRADED
        
        else:
            return ConnectionStatus.HEALTHY
    
    async def _check_alerts(self, metrics: ConnectionMetrics):
        """Check metrics against alert thresholds and log warnings."""
        alerts = []
        
        if metrics.connection_utilization >= self.alert_thresholds['utilization_critical']:
            alerts.append(f"CRITICAL: Connection utilization at {metrics.connection_utilization:.1%}")
        elif metrics.connection_utilization >= self.alert_thresholds['utilization_warning']:
            alerts.append(f"WARNING: Connection utilization at {metrics.connection_utilization:.1%}")
        
        if metrics.avg_query_time >= self.alert_thresholds['avg_query_time_critical']:
            alerts.append(f"CRITICAL: Average query time is {metrics.avg_query_time:.2f}s")
        elif metrics.avg_query_time >= self.alert_thresholds['avg_query_time_warning']:
            alerts.append(f"WARNING: Average query time is {metrics.avg_query_time:.2f}s")
        
        if metrics.failed_connections > 0:
            alerts.append(f"WARNING: {metrics.failed_connections} failed connections detected")
        
        for alert in alerts:
            if "CRITICAL" in alert:
                logger.critical(alert)
            else:
                logger.warning(alert)
    
    def get_current_metrics(self) -> Optional[ConnectionMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of metrics over a time window."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {"message": "No metrics in specified time window"}
        
        # Calculate aggregates
        avg_utilization = sum(m.connection_utilization for m in recent_metrics) / len(recent_metrics)
        max_utilization = max(m.connection_utilization for m in recent_metrics)
        avg_query_time = sum(m.avg_query_time for m in recent_metrics) / len(recent_metrics)
        max_query_time = max(m.avg_query_time for m in recent_metrics)
        total_failed = sum(m.failed_connections for m in recent_metrics)
        
        # Status distribution
        status_counts = {}
        for m in recent_metrics:
            status_counts[m.pool_status.value] = status_counts.get(m.pool_status.value, 0) + 1
        
        latest = recent_metrics[-1]
        
        return {
            "time_window_minutes": time_window_minutes,
            "metrics_count": len(recent_metrics),
            "current_status": latest.pool_status.value,
            "current_connections": {
                "active": latest.active_connections,
                "idle": latest.idle_connections,
                "total": latest.total_connections,
                "utilization": latest.connection_utilization
            },
            "performance_summary": {
                "avg_utilization": avg_utilization,
                "max_utilization": max_utilization,
                "avg_query_time": avg_query_time,
                "max_query_time": max_query_time,
                "total_failed_connections": total_failed
            },
            "status_distribution": status_counts,
            "last_check": self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    async def optimize_pool_configuration(self) -> Dict[str, Any]:
        """Analyze metrics and suggest pool configuration optimizations."""
        if not self.metrics_history:
            return {"message": "Insufficient data for optimization analysis"}
        
        # Analyze recent metrics (last hour)
        recent_metrics = [m for m in self.metrics_history 
                         if m.timestamp >= datetime.now() - timedelta(hours=1)]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        recommendations = []
        
        # Analyze utilization patterns
        utilizations = [m.connection_utilization for m in recent_metrics]
        avg_utilization = sum(utilizations) / len(utilizations)
        max_utilization = max(utilizations)
        
        if max_utilization > 0.95:
            recommendations.append({
                "type": "increase_max_connections",
                "reason": f"Peak utilization reached {max_utilization:.1%}",
                "suggestion": "Consider increasing max_connections by 25-50%"
            })
        elif avg_utilization < 0.3:
            recommendations.append({
                "type": "decrease_max_connections",
                "reason": f"Average utilization only {avg_utilization:.1%}",
                "suggestion": "Consider reducing max_connections to save resources"
            })
        
        # Analyze query performance
        query_times = [m.avg_query_time for m in recent_metrics if m.avg_query_time > 0]
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            
            if avg_query_time > 2.0:
                recommendations.append({
                    "type": "query_optimization",
                    "reason": f"Average query time is {avg_query_time:.2f}s",
                    "suggestion": "Consider query optimization or adding read replicas"
                })
        
        # Analyze connection failures
        total_failures = sum(m.failed_connections for m in recent_metrics)
        if total_failures > 0:
            failure_rate = total_failures / len(recent_metrics)
            recommendations.append({
                "type": "connection_reliability",
                "reason": f"Connection failure rate: {failure_rate:.2f} per minute",
                "suggestion": "Review connection timeout and retry settings"
            })
        
        # Current configuration (if available)
        current_config = {}
        if db_manager.is_postgresql():
            from mcp_server.database.postgres_database import postgres_db
            if postgres_db.pool:
                current_config = {
                    "current_max_connections": postgres_db.pool.get_max_size(),
                    "current_min_connections": postgres_db.pool.get_min_size()
                }
        
        return {
            "analysis_period": "last_hour",
            "metrics_analyzed": len(recent_metrics),
            "current_configuration": current_config,
            "performance_analysis": {
                "avg_utilization": avg_utilization,
                "max_utilization": max_utilization,
                "avg_query_time": sum(query_times) / len(query_times) if query_times else 0,
                "total_failures": total_failures
            },
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def run_connection_health_test(self) -> Dict[str, Any]:
        """Run comprehensive connection health test."""
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test 1: Basic connectivity
        try:
            start_time = time.time()
            if db_manager.is_postgresql():
                from mcp_server.database.postgres_database import postgres_db
                async with postgres_db.pool.acquire() as conn:
                    result = await conn.fetchval("SELECT 1")
                    test_results["tests"]["basic_connectivity"] = {
                        "status": "pass" if result == 1 else "fail",
                        "duration": time.time() - start_time,
                        "message": "Connection established successfully"
                    }
            else:
                from mcp_server.database.memory_database import get_db_connection
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    conn.close()
                    test_results["tests"]["basic_connectivity"] = {
                        "status": "pass" if result and result[0] == 1 else "fail",
                        "duration": time.time() - start_time,
                        "message": "SQLite connection established successfully"
                    }
                else:
                    test_results["tests"]["basic_connectivity"] = {
                        "status": "fail",
                        "duration": time.time() - start_time,
                        "message": "Failed to establish SQLite connection"
                    }
        except Exception as e:
            test_results["tests"]["basic_connectivity"] = {
                "status": "fail",
                "duration": time.time() - start_time,
                "message": f"Connection failed: {e}"
            }
        
        # Test 2: Query performance
        try:
            start_time = time.time()
            await db_manager.health_check()
            test_results["tests"]["health_check"] = {
                "status": "pass",
                "duration": time.time() - start_time,
                "message": "Health check completed successfully"
            }
        except Exception as e:
            test_results["tests"]["health_check"] = {
                "status": "fail",
                "duration": time.time() - start_time,
                "message": f"Health check failed: {e}"
            }
        
        # Test 3: Concurrent connections (PostgreSQL only)
        if db_manager.is_postgresql():
            try:
                start_time = time.time()
                from mcp_server.database.postgres_database import postgres_db
                
                # Test multiple concurrent connections
                tasks = []
                for i in range(5):
                    async def test_concurrent():
                        async with postgres_db.pool.acquire() as conn:
                            await asyncio.sleep(0.1)  # Simulate work
                            return await conn.fetchval("SELECT $1", i)
                    tasks.append(test_concurrent())
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                success_count = sum(1 for r in results if isinstance(r, int))
                test_results["tests"]["concurrent_connections"] = {
                    "status": "pass" if success_count == 5 else "partial",
                    "duration": time.time() - start_time,
                    "message": f"Completed {success_count}/5 concurrent connections",
                    "details": {"successful": success_count, "total": 5}
                }
            except Exception as e:
                test_results["tests"]["concurrent_connections"] = {
                    "status": "fail",
                    "duration": time.time() - start_time,
                    "message": f"Concurrent connection test failed: {e}"
                }
        
        # Overall status
        all_tests = test_results["tests"]
        passed_tests = sum(1 for test in all_tests.values() if test["status"] == "pass")
        total_tests = len(all_tests)
        
        test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "overall_status": "healthy" if passed_tests == total_tests else 
                            "degraded" if passed_tests > 0 else "critical"
        }
        
        return test_results
    
    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds."""
        self.alert_thresholds.update(thresholds)
        logger.info(f"Updated alert thresholds: {thresholds}")

# Global connection monitor instance
connection_monitor = DatabaseConnectionMonitor()

async def get_connection_monitor() -> DatabaseConnectionMonitor:
    """Get the connection monitor instance."""
    return connection_monitor