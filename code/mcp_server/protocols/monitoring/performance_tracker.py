"""
Performance Tracker for SIM-ONE Framework

This component provides comprehensive performance tracking and analysis capabilities
for all protocols in the SIM-ONE cognitive governance system. It monitors execution
times, resource utilization, throughput, and quality metrics while ensuring
adherence to the Five Laws of Cognitive Governance.

Key Features:
- Real-time protocol performance monitoring
- Statistical analysis and trending
- Resource utilization tracking per protocol
- Quality metrics and success rate monitoring
- Performance bottleneck identification
- Optimization recommendations
- Energy-efficient monitoring with adaptive sampling
- Historical performance analysis and reporting
- Integration with compliance and health monitoring
"""

import time
import logging
import threading
import statistics
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, NamedTuple, Tuple
from collections import deque, defaultdict, Counter
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import hashlib
import weakref
import gc

logger = logging.getLogger(__name__)

class PerformanceLevel(Enum):
    """Performance monitoring levels"""
    MINIMAL = "minimal"      # Basic metrics only
    STANDARD = "standard"    # Standard performance tracking
    DETAILED = "detailed"    # Comprehensive metrics with profiling
    DIAGNOSTIC = "diagnostic" # Full diagnostic with deep analysis

class MetricType(Enum):
    """Types of performance metrics"""
    TIMING = "timing"              # Execution time metrics
    THROUGHPUT = "throughput"      # Operations per unit time
    RESOURCE = "resource"          # CPU, memory, I/O usage
    QUALITY = "quality"            # Success rates, error metrics
    EFFICIENCY = "efficiency"      # Resource efficiency ratios
    LATENCY = "latency"           # Response time metrics

class PerformanceStatus(Enum):
    """Performance status levels"""
    EXCELLENT = "excellent"   # Performance exceeds expectations
    GOOD = "good"            # Performance meets requirements
    DEGRADED = "degraded"    # Performance below expectations
    CRITICAL = "critical"    # Performance critically impaired
    UNKNOWN = "unknown"      # Unable to assess performance

@dataclass
class TimingMetrics:
    """Detailed timing performance metrics"""
    total_executions: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    mean_execution_time: float = 0.0
    median_execution_time: float = 0.0
    std_execution_time: float = 0.0
    p95_execution_time: float = 0.0
    p99_execution_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, execution_time: float):
        """Update timing metrics with new execution"""
        self.total_executions += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        self.recent_times.append(execution_time)
        
        # Recalculate derived metrics
        if self.recent_times:
            times = list(self.recent_times)
            self.mean_execution_time = statistics.mean(times)
            self.median_execution_time = statistics.median(times)
            
            if len(times) > 1:
                self.std_execution_time = statistics.stdev(times)
                sorted_times = sorted(times)
                self.p95_execution_time = sorted_times[int(0.95 * len(sorted_times))]
                self.p99_execution_time = sorted_times[int(0.99 * len(sorted_times))]

@dataclass
class ThroughputMetrics:
    """Throughput performance metrics"""
    operations_per_second: float = 0.0
    operations_per_minute: float = 0.0
    peak_throughput: float = 0.0
    average_throughput: float = 0.0
    throughput_samples: deque = field(default_factory=lambda: deque(maxlen=300))  # 5 minutes at 1-second intervals
    last_calculation_time: float = field(default_factory=time.time)
    operations_since_last: int = 0
    
    def update(self, operations_count: int = 1):
        """Update throughput metrics"""
        current_time = time.time()
        self.operations_since_last += operations_count
        
        # Calculate throughput every second
        if current_time - self.last_calculation_time >= 1.0:
            time_delta = current_time - self.last_calculation_time
            current_ops_per_sec = self.operations_since_last / time_delta
            
            self.operations_per_second = current_ops_per_sec
            self.operations_per_minute = current_ops_per_sec * 60
            self.peak_throughput = max(self.peak_throughput, current_ops_per_sec)
            
            self.throughput_samples.append(current_ops_per_sec)
            
            if self.throughput_samples:
                self.average_throughput = statistics.mean(self.throughput_samples)
                
            self.operations_since_last = 0
            self.last_calculation_time = current_time

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    io_operations: int = 0
    network_bytes: int = 0
    file_operations: int = 0
    gc_collections: int = 0
    cpu_samples: deque = field(default_factory=lambda: deque(maxlen=60))
    memory_samples: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def update_cpu(self, cpu_percent: float):
        """Update CPU usage metrics"""
        self.cpu_usage_percent = cpu_percent
        self.cpu_samples.append(cpu_percent)
        
    def update_memory(self, memory_mb: float):
        """Update memory usage metrics"""
        self.memory_usage_mb = memory_mb
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.memory_samples.append(memory_mb)

@dataclass
class QualityMetrics:
    """Quality and reliability metrics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    success_rate: float = 0.0
    error_rate: float = 0.0
    error_types: Counter = field(default_factory=Counter)
    recent_results: deque = field(default_factory=lambda: deque(maxlen=1000))
    consecutive_failures: int = 0
    max_consecutive_failures: int = 0
    
    def update(self, success: bool, error_type: str = None):
        """Update quality metrics"""
        self.total_operations += 1
        
        if success:
            self.successful_operations += 1
            self.consecutive_failures = 0
        else:
            self.failed_operations += 1
            self.consecutive_failures += 1
            self.max_consecutive_failures = max(
                self.max_consecutive_failures, 
                self.consecutive_failures
            )
            
            if error_type:
                self.error_types[error_type] += 1
                
        self.recent_results.append(success)
        
        # Calculate rates
        if self.total_operations > 0:
            self.success_rate = self.successful_operations / self.total_operations
            self.error_rate = self.failed_operations / self.total_operations

@dataclass
class EfficiencyMetrics:
    """Efficiency and optimization metrics"""
    operations_per_cpu_percent: float = 0.0
    operations_per_mb_memory: float = 0.0
    energy_efficiency_score: float = 0.0
    resource_efficiency_score: float = 0.0
    cost_per_operation: float = 0.0
    optimization_opportunities: List[str] = field(default_factory=list)
    
    def calculate_efficiency(self, operations: int, cpu_usage: float, memory_mb: float):
        """Calculate efficiency metrics"""
        if cpu_usage > 0:
            self.operations_per_cpu_percent = operations / cpu_usage
            
        if memory_mb > 0:
            self.operations_per_mb_memory = operations / memory_mb
            
        # Energy efficiency score (0-100, higher is better)
        base_score = 100.0
        if cpu_usage > 80:
            base_score -= 30
        elif cpu_usage > 60:
            base_score -= 15
            
        if memory_mb > 1000:  # More than 1GB
            base_score -= 20
        elif memory_mb > 500:  # More than 500MB
            base_score -= 10
            
        self.energy_efficiency_score = max(0.0, base_score)
        
        # Resource efficiency score
        if operations > 0 and cpu_usage > 0 and memory_mb > 0:
            self.resource_efficiency_score = min(100.0, (operations / (cpu_usage * memory_mb)) * 1000)

class ProtocolPerformanceProfile:
    """Comprehensive performance profile for a protocol"""
    
    def __init__(self, protocol_name: str):
        self.protocol_name = protocol_name
        self.timing = TimingMetrics()
        self.throughput = ThroughputMetrics()
        self.resource = ResourceMetrics()
        self.quality = QualityMetrics()
        self.efficiency = EfficiencyMetrics()
        
        # Performance history
        self.performance_snapshots = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        
        # Metadata
        self.first_execution_time = time.time()
        self.last_execution_time = 0.0
        self.last_snapshot_time = 0.0
        
        # Performance thresholds (configurable per protocol)
        self.thresholds = {
            'max_execution_time': 5.0,  # seconds
            'min_success_rate': 0.95,   # 95%
            'max_cpu_usage': 10.0,      # percent
            'max_memory_mb': 100.0,     # MB
            'max_error_rate': 0.05      # 5%
        }
        
    def record_execution(self, execution_time: float, success: bool, 
                        cpu_usage: float = 0.0, memory_mb: float = 0.0,
                        error_type: str = None):
        """Record a protocol execution"""
        current_time = time.time()
        self.last_execution_time = current_time
        
        # Update all metrics
        self.timing.update(execution_time)
        self.throughput.update(1)  # One operation
        self.quality.update(success, error_type)
        
        if cpu_usage > 0:
            self.resource.update_cpu(cpu_usage)
        if memory_mb > 0:
            self.resource.update_memory(memory_mb)
            
        # Calculate efficiency
        self.efficiency.calculate_efficiency(
            self.quality.total_operations,
            self.resource.cpu_usage_percent,
            self.resource.memory_usage_mb
        )
        
        # Create periodic snapshots
        if current_time - self.last_snapshot_time >= 60.0:  # Every minute
            self._create_performance_snapshot()
            self.last_snapshot_time = current_time
            
    def _create_performance_snapshot(self):
        """Create a performance snapshot for historical analysis"""
        snapshot = {
            'timestamp': time.time(),
            'executions': self.timing.total_executions,
            'avg_execution_time': self.timing.mean_execution_time,
            'success_rate': self.quality.success_rate,
            'throughput': self.throughput.operations_per_second,
            'cpu_usage': self.resource.cpu_usage_percent,
            'memory_usage': self.resource.memory_usage_mb,
            'efficiency_score': self.efficiency.energy_efficiency_score
        }
        self.performance_snapshots.append(snapshot)
        
    def get_performance_status(self) -> PerformanceStatus:
        """Assess current performance status"""
        issues = []
        
        # Check execution time
        if self.timing.mean_execution_time > self.thresholds['max_execution_time']:
            issues.append("slow_execution")
            
        # Check success rate
        if self.quality.success_rate < self.thresholds['min_success_rate']:
            issues.append("low_success_rate")
            
        # Check resource usage
        if self.resource.cpu_usage_percent > self.thresholds['max_cpu_usage']:
            issues.append("high_cpu_usage")
            
        if self.resource.memory_usage_mb > self.thresholds['max_memory_mb']:
            issues.append("high_memory_usage")
            
        # Determine status based on issues
        if len(issues) >= 3:
            return PerformanceStatus.CRITICAL
        elif len(issues) == 2:
            return PerformanceStatus.DEGRADED
        elif len(issues) == 1:
            return PerformanceStatus.GOOD
        else:
            return PerformanceStatus.EXCELLENT
            
    def get_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Timing optimization
        if self.timing.std_execution_time > self.timing.mean_execution_time * 0.5:
            recommendations.append("Reduce execution time variability")
            
        # Throughput optimization
        if self.throughput.operations_per_second < 10:  # Less than 10 ops/sec
            recommendations.append("Optimize for higher throughput")
            
        # Resource optimization
        if self.resource.cpu_usage_percent > 20:
            recommendations.append("Optimize CPU usage")
            
        if self.resource.memory_usage_mb > 50:
            recommendations.append("Optimize memory usage")
            
        # Quality optimization
        if self.quality.error_rate > 0.01:  # More than 1% error rate
            recommendations.append("Improve error handling and reliability")
            
        return recommendations

class PerformanceTracker:
    """
    Comprehensive Performance Tracker for SIM-ONE Framework
    
    Provides real-time performance monitoring, analysis, and optimization
    recommendations for all protocols in the cognitive governance system.
    
    Features:
    - Multi-dimensional performance metrics (timing, throughput, quality, efficiency)
    - Statistical analysis and trending
    - Performance bottleneck identification
    - Optimization recommendations
    - Historical performance analysis
    - Energy-efficient monitoring
    - Integration with system health monitoring
    - Compliance with Five Laws of Cognitive Governance
    """
    
    def __init__(self,
                 monitoring_level: PerformanceLevel = PerformanceLevel.STANDARD,
                 collection_interval: float = 1.0,
                 max_protocols: int = 100,
                 performance_callback: Optional[Callable] = None):
        """
        Initialize Performance Tracker
        
        Args:
            monitoring_level: Level of performance monitoring detail
            collection_interval: Base interval for metric collection
            max_protocols: Maximum number of protocols to track
            performance_callback: Optional callback for performance events
        """
        self.monitoring_level = monitoring_level
        self.collection_interval = collection_interval
        self.max_protocols = max_protocols
        self.performance_callback = performance_callback
        
        # Protocol performance profiles
        self.protocol_profiles: Dict[str, ProtocolPerformanceProfile] = {}
        
        # System-wide performance metrics
        self.system_performance = {
            'total_operations': 0,
            'total_execution_time': 0.0,
            'average_throughput': 0.0,
            'system_efficiency_score': 0.0,
            'active_protocols': 0
        }
        
        # Performance trends
        self.performance_trends = deque(maxlen=1440)  # 24 hours of minute-level data
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self.lock = threading.RLock()
        
        # Performance thresholds
        self.global_thresholds = {
            'system_throughput_min': 50.0,      # ops/sec
            'system_efficiency_min': 70.0,      # score 0-100
            'protocol_count_max': 50,           # max active protocols
            'avg_response_time_max': 2.0,       # seconds
            'system_error_rate_max': 0.02       # 2%
        }
        
        # Optimization tracking
        self.optimization_history = []
        self.performance_alerts = deque(maxlen=1000)
        
        logger.info(f"PerformanceTracker initialized with level: {monitoring_level.value}")
        
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
            
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceTracker",
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Performance monitoring started")
        
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_monitoring:
            logger.warning("Performance monitoring is not running")
            return
            
        self.is_monitoring = False
        self.shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Performance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main performance monitoring loop"""
        logger.info("Performance monitoring loop started")
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Update system-wide performance metrics
                self._update_system_performance()
                
                # Analyze performance trends
                self._analyze_performance_trends()
                
                # Check for performance issues
                self._check_performance_thresholds()
                
                # Generate optimization recommendations
                self._generate_optimization_recommendations()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Calculate sleep time
                processing_time = time.time() - start_time
                sleep_time = max(0.1, self.collection_interval - processing_time)
                
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                self.shutdown_event.wait(1.0)
                
    def _update_system_performance(self):
        """Update system-wide performance metrics"""
        try:
            with self.lock:
                total_operations = sum(
                    profile.quality.total_operations 
                    for profile in self.protocol_profiles.values()
                )
                
                total_execution_time = sum(
                    profile.timing.total_execution_time 
                    for profile in self.protocol_profiles.values()
                )
                
                active_protocols = len([
                    p for p in self.protocol_profiles.values()
                    if time.time() - p.last_execution_time < 300  # Active in last 5 minutes
                ])
                
                # Calculate system throughput
                current_time = time.time()
                if hasattr(self, '_last_throughput_calculation'):
                    time_delta = current_time - self._last_throughput_calculation
                    if time_delta >= 60.0:  # Calculate every minute
                        operations_delta = total_operations - self.system_performance.get('last_total_operations', 0)
                        system_throughput = operations_delta / time_delta
                        
                        self.system_performance.update({
                            'total_operations': total_operations,
                            'total_execution_time': total_execution_time,
                            'average_throughput': system_throughput,
                            'active_protocols': active_protocols,
                            'last_total_operations': total_operations
                        })
                        
                        self._last_throughput_calculation = current_time
                else:
                    self._last_throughput_calculation = current_time
                    self.system_performance['last_total_operations'] = total_operations
                    
                # Calculate system efficiency score
                efficiency_scores = [
                    profile.efficiency.energy_efficiency_score
                    for profile in self.protocol_profiles.values()
                    if profile.efficiency.energy_efficiency_score > 0
                ]
                
                if efficiency_scores:
                    self.system_performance['system_efficiency_score'] = statistics.mean(efficiency_scores)
                    
        except Exception as e:
            logger.error(f"Error updating system performance: {e}")
            
    def _analyze_performance_trends(self):
        """Analyze performance trends and patterns"""
        try:
            current_time = time.time()
            
            # Create performance trend snapshot every minute
            if not hasattr(self, '_last_trend_time') or current_time - self._last_trend_time >= 60.0:
                trend_data = {
                    'timestamp': current_time,
                    'total_protocols': len(self.protocol_profiles),
                    'active_protocols': self.system_performance.get('active_protocols', 0),
                    'system_throughput': self.system_performance.get('average_throughput', 0.0),
                    'system_efficiency': self.system_performance.get('system_efficiency_score', 0.0),
                    'protocol_performances': {}
                }
                
                # Add individual protocol performances
                for name, profile in self.protocol_profiles.items():
                    if time.time() - profile.last_execution_time < 3600:  # Active in last hour
                        trend_data['protocol_performances'][name] = {
                            'success_rate': profile.quality.success_rate,
                            'avg_execution_time': profile.timing.mean_execution_time,
                            'throughput': profile.throughput.operations_per_second,
                            'efficiency': profile.efficiency.energy_efficiency_score
                        }
                        
                self.performance_trends.append(trend_data)
                self._last_trend_time = current_time
                
                # Analyze trends if we have enough data
                if len(self.performance_trends) >= 10:
                    self._detect_performance_patterns()
                    
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            
    def _detect_performance_patterns(self):
        """Detect performance patterns and anomalies"""
        try:
            recent_trends = list(self.performance_trends)[-10:]  # Last 10 minutes
            
            # Analyze system throughput trend
            throughputs = [t['system_throughput'] for t in recent_trends]
            if len(throughputs) >= 5:
                recent_avg = statistics.mean(throughputs[-5:])
                earlier_avg = statistics.mean(throughputs[:5])
                
                if recent_avg < earlier_avg * 0.8:  # 20% decrease
                    self._generate_performance_alert(
                        "throughput_decline",
                        "System throughput declining",
                        f"Throughput decreased from {earlier_avg:.1f} to {recent_avg:.1f} ops/sec"
                    )
                    
            # Analyze efficiency trend
            efficiencies = [t['system_efficiency'] for t in recent_trends if t['system_efficiency'] > 0]
            if len(efficiencies) >= 5:
                recent_efficiency = statistics.mean(efficiencies[-3:])
                if recent_efficiency < 60.0:  # Below 60% efficiency
                    self._generate_performance_alert(
                        "low_efficiency",
                        "Low system efficiency detected",
                        f"System efficiency at {recent_efficiency:.1f}%"
                    )
                    
        except Exception as e:
            logger.error(f"Error detecting performance patterns: {e}")
            
    def _check_performance_thresholds(self):
        """Check system performance against thresholds"""
        try:
            # Check system-wide thresholds
            system_perf = self.system_performance
            
            if system_perf.get('average_throughput', 0) < self.global_thresholds['system_throughput_min']:
                self._generate_performance_alert(
                    "low_system_throughput",
                    "System throughput below threshold",
                    f"Current: {system_perf.get('average_throughput', 0):.1f} ops/sec, "
                    f"Threshold: {self.global_thresholds['system_throughput_min']} ops/sec"
                )
                
            if system_perf.get('system_efficiency_score', 0) < self.global_thresholds['system_efficiency_min']:
                self._generate_performance_alert(
                    "low_system_efficiency",
                    "System efficiency below threshold",
                    f"Current: {system_perf.get('system_efficiency_score', 0):.1f}%, "
                    f"Threshold: {self.global_thresholds['system_efficiency_min']}%"
                )
                
            # Check individual protocol thresholds
            for name, profile in self.protocol_profiles.items():
                status = profile.get_performance_status()
                if status in [PerformanceStatus.DEGRADED, PerformanceStatus.CRITICAL]:
                    self._generate_performance_alert(
                        f"protocol_performance_{status.value}",
                        f"Protocol {name} performance {status.value}",
                        f"Success rate: {profile.quality.success_rate:.2%}, "
                        f"Avg time: {profile.timing.mean_execution_time:.3f}s"
                    )
                    
        except Exception as e:
            logger.error(f"Error checking performance thresholds: {e}")
            
    def _generate_optimization_recommendations(self):
        """Generate system-wide optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze protocol performance for system-wide recommendations
            slow_protocols = []
            resource_heavy_protocols = []
            unreliable_protocols = []
            
            for name, profile in self.protocol_profiles.items():
                if profile.timing.mean_execution_time > 1.0:  # More than 1 second
                    slow_protocols.append((name, profile.timing.mean_execution_time))
                    
                if profile.resource.memory_usage_mb > 100:  # More than 100MB
                    resource_heavy_protocols.append((name, profile.resource.memory_usage_mb))
                    
                if profile.quality.success_rate < 0.95:  # Less than 95% success
                    unreliable_protocols.append((name, profile.quality.success_rate))
                    
            # Generate recommendations based on analysis
            if slow_protocols:
                top_slow = sorted(slow_protocols, key=lambda x: x[1], reverse=True)[:3]
                recommendations.append(f"Optimize slow protocols: {', '.join([p[0] for p in top_slow])}")
                
            if resource_heavy_protocols:
                top_heavy = sorted(resource_heavy_protocols, key=lambda x: x[1], reverse=True)[:3]
                recommendations.append(f"Optimize memory usage in: {', '.join([p[0] for p in top_heavy])}")
                
            if unreliable_protocols:
                top_unreliable = sorted(unreliable_protocols, key=lambda x: x[1])[:3]
                recommendations.append(f"Improve reliability of: {', '.join([p[0] for p in top_unreliable])}")
                
            # System-level recommendations
            if self.system_performance.get('active_protocols', 0) > 20:
                recommendations.append("Consider consolidating or optimizing protocol usage")
                
            if self.system_performance.get('system_efficiency_score', 100) < 70:
                recommendations.append("System efficiency low - review resource allocation")
                
            # Store recommendations
            if recommendations:
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'recommendations': recommendations
                })
                
                # Keep only recent recommendations
                if len(self.optimization_history) > 100:
                    self.optimization_history.pop(0)
                    
        except Exception as e:
            logger.error(f"Error generating optimization recommendations: {e}")
            
    def _generate_performance_alert(self, alert_type: str, title: str, description: str):
        """Generate a performance alert"""
        try:
            alert = {
                'timestamp': time.time(),
                'type': alert_type,
                'title': title,
                'description': description
            }
            
            self.performance_alerts.append(alert)
            
            # Trigger callback if provided
            if self.performance_callback:
                try:
                    self.performance_callback(alert)
                except Exception as e:
                    logger.error(f"Error in performance callback: {e}")
                    
            logger.warning(f"Performance Alert: {title} - {description}")
            
        except Exception as e:
            logger.error(f"Error generating performance alert: {e}")
            
    def _cleanup_old_data(self):
        """Clean up old performance data"""
        try:
            current_time = time.time()
            cutoff_time = current_time - (24 * 3600)  # 24 hours ago
            
            # Clean up inactive protocol profiles
            inactive_protocols = []
            for name, profile in self.protocol_profiles.items():
                if profile.last_execution_time < cutoff_time:
                    inactive_protocols.append(name)
                    
            for name in inactive_protocols:
                if len(self.protocol_profiles) > self.max_protocols // 2:  # Only if we have many
                    del self.protocol_profiles[name]
                    logger.debug(f"Cleaned up inactive protocol profile: {name}")
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def register_protocol_execution(self, protocol_name: str, execution_time: float,
                                  success: bool, cpu_usage: float = 0.0,
                                  memory_mb: float = 0.0, error_type: str = None):
        """Register a protocol execution for performance tracking"""
        try:
            with self.lock:
                # Get or create protocol profile
                if protocol_name not in self.protocol_profiles:
                    if len(self.protocol_profiles) >= self.max_protocols:
                        # Remove oldest inactive protocol
                        oldest_protocol = min(
                            self.protocol_profiles.items(),
                            key=lambda x: x[1].last_execution_time
                        )[0]
                        del self.protocol_profiles[oldest_protocol]
                        
                    self.protocol_profiles[protocol_name] = ProtocolPerformanceProfile(protocol_name)
                    
                # Record execution
                profile = self.protocol_profiles[protocol_name]
                profile.record_execution(execution_time, success, cpu_usage, memory_mb, error_type)
                
        except Exception as e:
            logger.error(f"Error registering protocol execution for {protocol_name}: {e}")
            
    def get_protocol_performance(self, protocol_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific protocol"""
        try:
            with self.lock:
                profile = self.protocol_profiles.get(protocol_name)
                if not profile:
                    return {'error': f'Protocol {protocol_name} not found'}
                    
                return {
                    'protocol_name': protocol_name,
                    'status': profile.get_performance_status().value,
                    'timing': {
                        'total_executions': profile.timing.total_executions,
                        'mean_execution_time': profile.timing.mean_execution_time,
                        'median_execution_time': profile.timing.median_execution_time,
                        'p95_execution_time': profile.timing.p95_execution_time,
                        'p99_execution_time': profile.timing.p99_execution_time,
                        'std_deviation': profile.timing.std_execution_time
                    },
                    'throughput': {
                        'operations_per_second': profile.throughput.operations_per_second,
                        'operations_per_minute': profile.throughput.operations_per_minute,
                        'peak_throughput': profile.throughput.peak_throughput,
                        'average_throughput': profile.throughput.average_throughput
                    },
                    'quality': {
                        'success_rate': profile.quality.success_rate,
                        'error_rate': profile.quality.error_rate,
                        'total_operations': profile.quality.total_operations,
                        'consecutive_failures': profile.quality.consecutive_failures,
                        'error_types': dict(profile.quality.error_types)
                    },
                    'resource': {
                        'cpu_usage_percent': profile.resource.cpu_usage_percent,
                        'memory_usage_mb': profile.resource.memory_usage_mb,
                        'peak_memory_mb': profile.resource.peak_memory_mb
                    },
                    'efficiency': {
                        'energy_efficiency_score': profile.efficiency.energy_efficiency_score,
                        'operations_per_cpu_percent': profile.efficiency.operations_per_cpu_percent,
                        'operations_per_mb_memory': profile.efficiency.operations_per_mb_memory
                    },
                    'recommendations': profile.get_optimization_recommendations(),
                    'thresholds': profile.thresholds,
                    'first_execution': profile.first_execution_time,
                    'last_execution': profile.last_execution_time
                }
                
        except Exception as e:
            logger.error(f"Error getting protocol performance for {protocol_name}: {e}")
            return {'error': str(e)}
            
    def get_system_performance(self) -> Dict[str, Any]:
        """Get system-wide performance metrics"""
        try:
            with self.lock:
                protocol_summary = {}
                for name, profile in self.protocol_profiles.items():
                    protocol_summary[name] = {
                        'status': profile.get_performance_status().value,
                        'executions': profile.timing.total_executions,
                        'success_rate': profile.quality.success_rate,
                        'avg_time': profile.timing.mean_execution_time,
                        'throughput': profile.throughput.operations_per_second
                    }
                    
                return {
                    'system_metrics': self.system_performance,
                    'protocol_count': len(self.protocol_profiles),
                    'active_protocols': len([
                        p for p in self.protocol_profiles.values()
                        if time.time() - p.last_execution_time < 300
                    ]),
                    'monitoring_level': self.monitoring_level.value,
                    'thresholds': self.global_thresholds,
                    'recent_alerts': list(self.performance_alerts)[-10:],
                    'recent_recommendations': (
                        self.optimization_history[-1]['recommendations']
                        if self.optimization_history else []
                    ),
                    'protocol_summary': protocol_summary
                }
                
        except Exception as e:
            logger.error(f"Error getting system performance: {e}")
            return {'error': str(e)}
            
    def get_performance_trends(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get performance trends for specified time period"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self.lock:
                recent_trends = [
                    trend for trend in self.performance_trends
                    if trend['timestamp'] >= cutoff_time
                ]
                
                return {
                    'time_period_hours': hours,
                    'data_points': len(recent_trends),
                    'trends': recent_trends
                }
                
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == '__main__':
    import random
    import signal
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def performance_alert_handler(alert):
        """Handle performance alerts"""
        print(f"⚠️  PERFORMANCE ALERT: {alert['title']}")
        print(f"   Type: {alert['type']}")
        print(f"   Description: {alert['description']}")
        print()
        
    # Create performance tracker
    tracker = PerformanceTracker(
        monitoring_level=PerformanceLevel.DETAILED,
        performance_callback=performance_alert_handler
    )
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down performance tracker...")
        tracker.stop_monitoring()
        sys.exit(0)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start monitoring
        print("📊 Starting Performance Tracker")
        print(f"   Monitoring Level: {tracker.monitoring_level.value}")
        print("   Press Ctrl+C to stop\n")
        
        tracker.start_monitoring()
        
        # Simulate protocol executions
        protocol_names = ["REP", "HIP", "POCP", "EEP", "VVP", "CCP", "ESL", "MTP", "SP"]
        
        for i in range(100):
            time.sleep(1.0)
            
            # Simulate multiple protocol executions per second
            for _ in range(random.randint(1, 5)):
                protocol = random.choice(protocol_names)
                execution_time = random.uniform(0.01, 2.0)
                success = random.random() > 0.05  # 95% success rate
                cpu_usage = random.uniform(0.5, 15.0)
                memory_mb = random.uniform(5.0, 50.0)
                error_type = random.choice(["timeout", "validation", "resource"]) if not success else None
                
                tracker.register_protocol_execution(
                    protocol_name=protocol,
                    execution_time=execution_time,
                    success=success,
                    cpu_usage=cpu_usage,
                    memory_mb=memory_mb,
                    error_type=error_type
                )
                
            # Print status every 10 seconds
            if (i + 1) % 10 == 0:
                system_perf = tracker.get_system_performance()
                print(f"📈 Performance Update (Iteration {i + 1}):")
                print(f"   Total Protocols: {system_perf['protocol_count']}")
                print(f"   Active Protocols: {system_perf['active_protocols']}")
                print(f"   System Throughput: {system_perf['system_metrics'].get('average_throughput', 0):.1f} ops/sec")
                print(f"   System Efficiency: {system_perf['system_metrics'].get('system_efficiency_score', 0):.1f}%")
                
                # Show top performing protocol
                if system_perf['protocol_summary']:
                    best_protocol = max(
                        system_perf['protocol_summary'].items(),
                        key=lambda x: x[1]['success_rate']
                    )
                    print(f"   Best Protocol: {best_protocol[0]} ({best_protocol[1]['success_rate']:.2%} success)")
                print()
                
        # Show detailed performance analysis
        print("\n📊 Detailed Performance Analysis:")
        for protocol in protocol_names[:3]:  # Show first 3 protocols
            perf = tracker.get_protocol_performance(protocol)
            if 'error' not in perf:
                print(f"\n{protocol} Protocol Performance:")
                print(f"   Status: {perf['status'].upper()}")
                print(f"   Executions: {perf['timing']['total_executions']}")
                print(f"   Success Rate: {perf['quality']['success_rate']:.2%}")
                print(f"   Avg Execution Time: {perf['timing']['mean_execution_time']:.3f}s")
                print(f"   Throughput: {perf['throughput']['operations_per_second']:.1f} ops/sec")
                print(f"   Efficiency Score: {perf['efficiency']['energy_efficiency_score']:.1f}")
                
                if perf['recommendations']:
                    print(f"   Recommendations: {', '.join(perf['recommendations'])}")
                    
        # Keep running until interrupted
        print("\n✅ Performance tracking active. Press Ctrl+C to stop...")
        while tracker.is_monitoring:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        tracker.stop_monitoring()
        print("Performance Tracker demonstration completed.")