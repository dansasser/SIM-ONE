"""
Real-Time Monitoring Protocol for SIM-ONE Framework

This protocol implements comprehensive real-time monitoring capabilities that provide
continuous oversight of the entire SIM-ONE cognitive governance system while adhering
to the Five Laws of Cognitive Governance.

Architecture:
- Law 1 (Architectural Intelligence): Monitoring emerges from protocol coordination
- Law 2 (Cognitive Governance): All monitoring processes are governed by specialized protocols  
- Law 3 (Truth Foundation): All monitoring data is grounded in absolute truth principles
- Law 4 (Energy Stewardship): Minimal computational overhead through efficient design
- Law 5 (Deterministic Reliability): Consistent, predictable monitoring behavior

The protocol provides:
- Real-time system health monitoring
- Protocol performance tracking
- Five Laws compliance monitoring  
- Resource utilization analysis
- Error detection and classification
- Performance bottleneck identification
- Predictive failure analysis
- Multi-level alerting system
- Historical trend analysis
- Automated recovery triggers
"""

import asyncio
import logging
import json
import time
import threading
import queue
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, NamedTuple, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
import gc
import sys
import tracemalloc
import resource
import signal
import os
from pathlib import Path

# Import SIM-ONE core components
from ..governance.five_laws_validator.law1_architectural_intelligence import Law1ArchitecturalIntelligence
from ..governance.five_laws_validator.law2_cognitive_governance import Law2CognitiveGovernance
from ..governance.five_laws_validator.law3_truth_foundation import Law3TruthFoundation
from ..governance.five_laws_validator.law4_energy_stewardship import Law4EnergyStewardship
from ..governance.five_laws_validator.law5_deterministic_reliability import Law5DeterministicReliability

logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring intensity levels for adaptive resource management"""
    MINIMAL = "minimal"          # Critical alerts only, <1% CPU overhead
    STANDARD = "standard"        # Standard monitoring, <2% CPU overhead  
    DETAILED = "detailed"        # Comprehensive monitoring, <5% CPU overhead
    DIAGNOSTIC = "diagnostic"    # Full diagnostic mode, <10% CPU overhead

class AlertSeverity(Enum):
    """Alert severity levels following incident response best practices"""
    CRITICAL = "critical"        # System-threatening issues requiring immediate action
    HIGH = "high"               # Significant issues requiring urgent attention
    MEDIUM = "medium"           # Important issues requiring timely resolution
    LOW = "low"                 # Minor issues for awareness and tracking
    INFO = "info"               # Informational notifications

class MonitoringMetric(NamedTuple):
    """Standard structure for all monitoring metrics"""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str]
    source: str

class AlertEvent(NamedTuple):
    """Standard structure for alert events"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: float
    source: str
    tags: Dict[str, str]
    suggested_actions: List[str]
    auto_recovery_possible: bool

@dataclass
class ProtocolPerformanceData:
    """Performance data for individual protocols"""
    protocol_name: str
    execution_count: int = 0
    total_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    max_execution_time: float = 0.0
    error_count: int = 0
    success_rate: float = 0.0
    average_memory_usage: float = 0.0
    cpu_usage_samples: List[float] = field(default_factory=list)
    recent_errors: deque = field(default_factory=lambda: deque(maxlen=100))
    compliance_violations: Dict[str, int] = field(default_factory=dict)
    
    def update_execution(self, execution_time: float, success: bool, memory_usage: float = 0.0):
        """Update performance data with new execution"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.min_execution_time = min(self.min_execution_time, execution_time)
        self.max_execution_time = max(self.max_execution_time, execution_time)
        
        if not success:
            self.error_count += 1
            
        self.success_rate = (self.execution_count - self.error_count) / self.execution_count
        
        if memory_usage > 0:
            # Moving average for memory usage
            if hasattr(self, '_memory_samples'):
                self._memory_samples.append(memory_usage)
                if len(self._memory_samples) > 1000:  # Keep last 1000 samples
                    self._memory_samples.pop(0)
            else:
                self._memory_samples = [memory_usage]
            self.average_memory_usage = statistics.mean(self._memory_samples)

@dataclass  
class SystemHealthData:
    """Comprehensive system health information"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    open_files: int = 0
    active_threads: int = 0
    gc_collections: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    system_load: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

@dataclass
class FiveLawsComplianceStatus:
    """Compliance status for each of the Five Laws"""
    law1_score: float = 0.0  # Architectural Intelligence compliance
    law2_score: float = 0.0  # Cognitive Governance compliance  
    law3_score: float = 0.0  # Truth Foundation compliance
    law4_score: float = 0.0  # Energy Stewardship compliance
    law5_score: float = 0.0  # Deterministic Reliability compliance
    overall_score: float = 0.0
    violations: Dict[str, List[str]] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def calculate_overall_score(self):
        """Calculate overall compliance score"""
        scores = [self.law1_score, self.law2_score, self.law3_score, 
                 self.law4_score, self.law5_score]
        self.overall_score = statistics.mean(scores) if scores else 0.0

class MonitoringDataBuffer:
    """Thread-safe circular buffer for monitoring data"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.lock = threading.RLock()
        
    def append(self, item):
        """Thread-safe append to buffer"""
        with self.lock:
            self.data.append(item)
            
    def get_recent(self, count: int) -> List:
        """Get recent items from buffer"""
        with self.lock:
            return list(self.data)[-count:] if count <= len(self.data) else list(self.data)
            
    def get_timerange(self, start_time: float, end_time: float) -> List:
        """Get items within specified time range"""
        with self.lock:
            return [item for item in self.data 
                   if hasattr(item, 'timestamp') and start_time <= item.timestamp <= end_time]

class RealTimeMonitorProtocol:
    """
    Comprehensive Real-Time Monitoring Protocol for SIM-ONE Framework
    
    This protocol implements continuous monitoring capabilities that provide real-time
    oversight of the entire cognitive governance system while maintaining adherence
    to the Five Laws of Cognitive Governance.
    
    Key Features:
    - Real-time system health monitoring with sub-second latency
    - Protocol performance tracking with detailed metrics
    - Five Laws compliance monitoring and violation detection
    - Resource utilization analysis and optimization recommendations
    - Predictive failure analysis using historical data patterns
    - Multi-level alerting system with intelligent escalation
    - Automated recovery triggers for common failure scenarios
    - Energy-efficient monitoring with adaptive resource allocation
    
    Architecture:
    The protocol uses a multi-threaded architecture with separate threads for:
    - System metrics collection (high frequency, low latency)
    - Protocol performance monitoring (event-driven)
    - Compliance validation (scheduled intervals)
    - Alert processing and notification (asynchronous)
    - Data aggregation and analysis (background processing)
    """
    
    def __init__(self, 
                 monitoring_level: MonitoringLevel = MonitoringLevel.STANDARD,
                 collection_interval: float = 1.0,
                 alert_callback: Optional[Callable] = None,
                 data_retention_hours: int = 24,
                 enable_predictive_analysis: bool = True):
        """
        Initialize the Real-Time Monitoring Protocol
        
        Args:
            monitoring_level: Intensity level for monitoring operations
            collection_interval: Base interval for metric collection (seconds)
            alert_callback: Optional callback for alert notifications
            data_retention_hours: Hours to retain monitoring data
            enable_predictive_analysis: Enable predictive failure analysis
        """
        self.monitoring_level = monitoring_level
        self.collection_interval = collection_interval
        self.alert_callback = alert_callback
        self.data_retention_hours = data_retention_hours
        self.enable_predictive_analysis = enable_predictive_analysis
        
        # Initialize core components
        self._initialize_validators()
        self._initialize_data_structures()
        self._initialize_threading()
        self._initialize_metrics()
        
        # State management
        self.is_running = False
        self.start_time = time.time()
        self.last_health_check = 0.0
        
        # Performance optimization
        self._setup_performance_optimization()
        
        logger.info(f"RealTimeMonitorProtocol initialized with level: {monitoring_level.value}")
        
    def _initialize_validators(self):
        """Initialize Five Laws validators for compliance monitoring"""
        try:
            self.law1_validator = Law1ArchitecturalIntelligence()
            self.law2_validator = Law2CognitiveGovernance()  
            self.law3_validator = Law3TruthFoundation()
            self.law4_validator = Law4EnergyStewardship()
            self.law5_validator = Law5DeterministicReliability()
            logger.info("Five Laws validators initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Five Laws validators: {e}")
            # Fallback to mock validators for graceful degradation
            self._initialize_fallback_validators()
            
    def _initialize_fallback_validators(self):
        """Initialize fallback validators if main validators fail"""
        class FallbackValidator:
            def validate(self, *args, **kwargs):
                return {"compliant": True, "score": 0.0, "violations": []}
                
        self.law1_validator = FallbackValidator()
        self.law2_validator = FallbackValidator()
        self.law3_validator = FallbackValidator()
        self.law4_validator = FallbackValidator()
        self.law5_validator = FallbackValidator()
        logger.warning("Using fallback validators due to initialization failure")
        
    def _initialize_data_structures(self):
        """Initialize data structures for monitoring"""
        # Core data buffers
        self.metrics_buffer = MonitoringDataBuffer(max_size=50000)
        self.alerts_buffer = MonitoringDataBuffer(max_size=10000)
        self.performance_buffer = MonitoringDataBuffer(max_size=25000)
        
        # Protocol performance tracking
        self.protocol_performance: Dict[str, ProtocolPerformanceData] = {}
        
        # System health tracking
        self.system_health_history = deque(maxlen=3600)  # 1 hour at 1-second intervals
        self.current_system_health = SystemHealthData()
        
        # Compliance tracking
        self.compliance_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.current_compliance = FiveLawsComplianceStatus()
        
        # Alert management
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_suppression: Dict[str, float] = {}  # Alert type -> suppression end time
        
        # Performance thresholds (configurable)
        self.performance_thresholds = {
            'cpu_usage_critical': 90.0,
            'cpu_usage_warning': 75.0,
            'memory_usage_critical': 95.0,
            'memory_usage_warning': 80.0,
            'disk_usage_critical': 95.0,
            'disk_usage_warning': 85.0,
            'protocol_error_rate_critical': 10.0,
            'protocol_error_rate_warning': 5.0,
            'response_time_critical': 5.0,
            'response_time_warning': 2.0,
        }
        
        logger.info("Data structures initialized successfully")
        
    def _initialize_threading(self):
        """Initialize threading components"""
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="RealTimeMonitor")
        self.data_queue = queue.Queue(maxsize=1000)
        self.alert_queue = queue.Queue(maxsize=500)
        
        # Thread control
        self.shutdown_event = threading.Event()
        self.threads: Dict[str, threading.Thread] = {}
        
        logger.info("Threading components initialized")
        
    def _initialize_metrics(self):
        """Initialize metrics collection system"""
        # Enable memory tracking for diagnostics
        if self.monitoring_level == MonitoringLevel.DIAGNOSTIC:
            tracemalloc.start()
            
        # System metrics collector setup
        self.system_metrics = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'disk_total': psutil.disk_usage('/').total,
            'boot_time': psutil.boot_time(),
            'process_start_time': time.time()
        }
        
        logger.info("Metrics collection initialized")
        
    def _setup_performance_optimization(self):
        """Setup performance optimization based on monitoring level"""
        level_configs = {
            MonitoringLevel.MINIMAL: {
                'collection_interval': 5.0,
                'batch_size': 100,
                'enable_detailed_metrics': False,
                'enable_memory_profiling': False
            },
            MonitoringLevel.STANDARD: {
                'collection_interval': 1.0,
                'batch_size': 50,
                'enable_detailed_metrics': True,
                'enable_memory_profiling': False
            },
            MonitoringLevel.DETAILED: {
                'collection_interval': 0.5,
                'batch_size': 25,
                'enable_detailed_metrics': True,
                'enable_memory_profiling': True
            },
            MonitoringLevel.DIAGNOSTIC: {
                'collection_interval': 0.1,
                'batch_size': 10,
                'enable_detailed_metrics': True,
                'enable_memory_profiling': True
            }
        }
        
        config = level_configs.get(self.monitoring_level, level_configs[MonitoringLevel.STANDARD])
        self.effective_collection_interval = config['collection_interval']
        self.batch_size = config['batch_size']
        self.enable_detailed_metrics = config['enable_detailed_metrics']
        self.enable_memory_profiling = config['enable_memory_profiling']
        
        logger.info(f"Performance optimization configured for {self.monitoring_level.value}")
        
    def start_monitoring(self):
        """Start the real-time monitoring system"""
        if self.is_running:
            logger.warning("Monitoring system is already running")
            return
            
        self.is_running = True
        self.start_time = time.time()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        logger.info("Real-time monitoring system started successfully")
        
    def _start_monitoring_threads(self):
        """Start all monitoring threads"""
        thread_configs = [
            ('system_metrics', self._system_metrics_collector_thread),
            ('compliance_monitor', self._compliance_monitoring_thread),
            ('alert_processor', self._alert_processing_thread),
            ('data_aggregator', self._data_aggregation_thread),
        ]
        
        if self.enable_predictive_analysis:
            thread_configs.append(('predictive_analyzer', self._predictive_analysis_thread))
            
        for thread_name, thread_func in thread_configs:
            thread = threading.Thread(
                target=thread_func,
                name=f"RealTimeMonitor-{thread_name}",
                daemon=True
            )
            thread.start()
            self.threads[thread_name] = thread
            logger.info(f"Started {thread_name} thread")
            
    def _system_metrics_collector_thread(self):
        """System metrics collection thread - high frequency monitoring"""
        logger.info("System metrics collector thread started")
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Collect system health data
                health_data = self._collect_system_health()
                
                # Store in buffer
                self.current_system_health = health_data
                self.system_health_history.append(health_data)
                
                # Check for threshold violations
                self._check_system_health_thresholds(health_data)
                
                # Adaptive sleep based on monitoring level and CPU usage
                collection_time = time.time() - start_time
                sleep_time = max(0.1, self.effective_collection_interval - collection_time)
                
                # Adjust sleep time based on system load (energy stewardship)
                if health_data.cpu_usage > 80:
                    sleep_time *= 1.5  # Reduce monitoring frequency under high load
                    
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in system metrics collection: {e}")
                self.shutdown_event.wait(1.0)  # Brief pause before retry
                
    def _collect_system_health(self) -> SystemHealthData:
        """Collect comprehensive system health data"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Process information
            current_process = psutil.Process()
            open_files = len(current_process.open_files())
            active_threads = current_process.num_threads()
            
            # Garbage collection stats
            gc_stats = {}
            if hasattr(gc, 'get_stats'):
                for i, stats in enumerate(gc.get_stats()):
                    gc_stats[f'gen_{i}_collections'] = stats['collections']
                    
            # System load
            system_load = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemHealthData(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                open_files=open_files,
                active_threads=active_threads,
                gc_collections=gc_stats,
                process_count=process_count,
                system_load=system_load,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system health data: {e}")
            return SystemHealthData()  # Return default values
            
    def _check_system_health_thresholds(self, health_data: SystemHealthData):
        """Check system health against thresholds and generate alerts"""
        current_time = time.time()
        
        # CPU usage alerts
        if health_data.cpu_usage > self.performance_thresholds['cpu_usage_critical']:
            self._generate_alert(
                alert_id=f"cpu_critical_{int(current_time)}",
                severity=AlertSeverity.CRITICAL,
                title="Critical CPU Usage",
                description=f"CPU usage is {health_data.cpu_usage:.1f}%, exceeding critical threshold",
                source="system_health",
                tags={"metric": "cpu_usage", "value": str(health_data.cpu_usage)},
                suggested_actions=[
                    "Identify high-CPU processes",
                    "Scale resources if possible",
                    "Reduce monitoring frequency temporarily"
                ],
                auto_recovery_possible=True
            )
        elif health_data.cpu_usage > self.performance_thresholds['cpu_usage_warning']:
            self._generate_alert(
                alert_id=f"cpu_warning_{int(current_time)}",
                severity=AlertSeverity.MEDIUM,
                title="High CPU Usage",
                description=f"CPU usage is {health_data.cpu_usage:.1f}%, exceeding warning threshold",
                source="system_health", 
                tags={"metric": "cpu_usage", "value": str(health_data.cpu_usage)},
                suggested_actions=["Monitor CPU trends", "Consider resource optimization"],
                auto_recovery_possible=False
            )
            
        # Memory usage alerts
        if health_data.memory_usage > self.performance_thresholds['memory_usage_critical']:
            self._generate_alert(
                alert_id=f"memory_critical_{int(current_time)}",
                severity=AlertSeverity.CRITICAL,
                title="Critical Memory Usage",
                description=f"Memory usage is {health_data.memory_usage:.1f}%, exceeding critical threshold",
                source="system_health",
                tags={"metric": "memory_usage", "value": str(health_data.memory_usage)},
                suggested_actions=[
                    "Force garbage collection",
                    "Clear monitoring buffers", 
                    "Restart system if necessary"
                ],
                auto_recovery_possible=True
            )
            
        # Disk usage alerts
        if health_data.disk_usage > self.performance_thresholds['disk_usage_critical']:
            self._generate_alert(
                alert_id=f"disk_critical_{int(current_time)}",
                severity=AlertSeverity.HIGH,
                title="Critical Disk Usage",
                description=f"Disk usage is {health_data.disk_usage:.1f}%, exceeding critical threshold",
                source="system_health",
                tags={"metric": "disk_usage", "value": str(health_data.disk_usage)},
                suggested_actions=[
                    "Clean temporary files",
                    "Archive old monitoring data",
                    "Expand storage capacity"
                ],
                auto_recovery_possible=True
            )
            
    def _compliance_monitoring_thread(self):
        """Compliance monitoring thread - validates Five Laws adherence"""
        logger.info("Compliance monitoring thread started")
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Collect compliance data
                compliance_status = self._assess_five_laws_compliance()
                
                # Store compliance data
                self.current_compliance = compliance_status
                self.compliance_history.append(compliance_status)
                
                # Check for compliance violations
                self._check_compliance_violations(compliance_status)
                
                # Sleep interval for compliance checking (typically longer than system metrics)
                compliance_interval = self.effective_collection_interval * 10  # 10x system metrics interval
                collection_time = time.time() - start_time
                sleep_time = max(1.0, compliance_interval - collection_time)
                
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in compliance monitoring: {e}")
                self.shutdown_event.wait(5.0)
                
    def _assess_five_laws_compliance(self) -> FiveLawsComplianceStatus:
        """Assess compliance with the Five Laws of Cognitive Governance"""
        compliance_status = FiveLawsComplianceStatus()
        
        try:
            # Gather system data for compliance assessment
            system_data = {
                'current_health': self.current_system_health,
                'protocol_performance': dict(self.protocol_performance),
                'monitoring_overhead': self._calculate_monitoring_overhead(),
                'recent_metrics': self.metrics_buffer.get_recent(100)
            }
            
            # Law 1: Architectural Intelligence
            # Intelligence emerges from coordination, not scale
            law1_result = self._assess_law1_compliance(system_data)
            compliance_status.law1_score = law1_result.get('score', 0.0)
            
            # Law 2: Cognitive Governance  
            # Every process must be governed by specialized protocols
            law2_result = self._assess_law2_compliance(system_data)
            compliance_status.law2_score = law2_result.get('score', 0.0)
            
            # Law 3: Truth Foundation
            # All reasoning grounded in absolute truth principles
            law3_result = self._assess_law3_compliance(system_data)
            compliance_status.law3_score = law3_result.get('score', 0.0)
            
            # Law 4: Energy Stewardship
            # Maximum intelligence with minimal computational resources
            law4_result = self._assess_law4_compliance(system_data)
            compliance_status.law4_score = law4_result.get('score', 0.0)
            
            # Law 5: Deterministic Reliability
            # Consistent, predictable outcomes
            law5_result = self._assess_law5_compliance(system_data)
            compliance_status.law5_score = law5_result.get('score', 0.0)
            
            # Compile violations
            compliance_status.violations = {
                'law1': law1_result.get('violations', []),
                'law2': law2_result.get('violations', []),
                'law3': law3_result.get('violations', []),
                'law4': law4_result.get('violations', []),
                'law5': law5_result.get('violations', [])
            }
            
            # Calculate overall compliance score
            compliance_status.calculate_overall_score()
            
        except Exception as e:
            logger.error(f"Error assessing Five Laws compliance: {e}")
            # Set default scores on error
            compliance_status.law1_score = 0.0
            compliance_status.law2_score = 0.0
            compliance_status.law3_score = 0.0
            compliance_status.law4_score = 0.0
            compliance_status.law5_score = 0.0
            compliance_status.overall_score = 0.0
            
        return compliance_status
        
    def _assess_law1_compliance(self, system_data: Dict) -> Dict[str, Any]:
        """Assess Law 1: Architectural Intelligence compliance"""
        violations = []
        score = 100.0
        
        try:
            # Check if intelligence emerges from protocol coordination
            protocol_count = len(system_data['protocol_performance'])
            
            if protocol_count < 3:
                violations.append("Insufficient protocol coordination for architectural intelligence")
                score -= 30.0
                
            # Check protocol interaction efficiency
            total_protocols = len(system_data['protocol_performance'])
            successful_protocols = sum(1 for p in system_data['protocol_performance'].values() 
                                     if p.success_rate > 0.9)
            
            coordination_efficiency = successful_protocols / total_protocols if total_protocols > 0 else 0.0
            
            if coordination_efficiency < 0.8:
                violations.append(f"Protocol coordination efficiency too low: {coordination_efficiency:.2%}")
                score -= 25.0
                
            # Check for architectural patterns vs brute force approaches
            avg_execution_time = statistics.mean([
                p.total_execution_time / max(p.execution_count, 1)
                for p in system_data['protocol_performance'].values()
            ]) if system_data['protocol_performance'] else 0.0
            
            if avg_execution_time > 1.0:  # More than 1 second average
                violations.append(f"Protocol execution times suggest non-architectural approach: {avg_execution_time:.3f}s avg")
                score -= 20.0
                
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing Law 1 compliance: {e}")
            score = 0.0
            violations.append(f"Assessment error: {str(e)}")
            
        return {'score': score, 'violations': violations}
        
    def _assess_law2_compliance(self, system_data: Dict) -> Dict[str, Any]:
        """Assess Law 2: Cognitive Governance compliance"""
        violations = []
        score = 100.0
        
        try:
            # Check that all processes have governance protocols
            ungoverned_processes = []
            
            # Verify monitoring itself is governed (meta-governance)
            if not hasattr(self, 'law1_validator'):
                violations.append("Monitoring system lacks governance validation")
                score -= 40.0
                
            # Check protocol error handling governance
            protocols_with_poor_governance = []
            for name, perf in system_data['protocol_performance'].items():
                if perf.error_count > 0 and len(perf.recent_errors) == 0:
                    protocols_with_poor_governance.append(name)
                    
            if protocols_with_poor_governance:
                violations.append(f"Protocols lack proper error governance: {protocols_with_poor_governance}")
                score -= 15.0 * len(protocols_with_poor_governance)
                
            # Check for specialized protocol governance
            required_governance_types = ['validation', 'error_recovery', 'performance_monitoring']
            # This would typically check against a registry of governance protocols
            
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing Law 2 compliance: {e}")
            score = 0.0
            violations.append(f"Assessment error: {str(e)}")
            
        return {'score': score, 'violations': violations}
        
    def _assess_law3_compliance(self, system_data: Dict) -> Dict[str, Any]:
        """Assess Law 3: Truth Foundation compliance"""
        violations = []
        score = 100.0
        
        try:
            # Check for absolute truth principles in monitoring data
            
            # Verify data integrity - no NaN or invalid values
            invalid_metrics = 0
            total_metrics = 0
            
            for metric in system_data['recent_metrics']:
                total_metrics += 1
                if hasattr(metric, 'value'):
                    if not isinstance(metric.value, (int, float)) or metric.value != metric.value:  # NaN check
                        invalid_metrics += 1
                        
            if total_metrics > 0 and invalid_metrics / total_metrics > 0.01:  # >1% invalid
                violations.append(f"High rate of invalid metrics: {invalid_metrics}/{total_metrics}")
                score -= 30.0
                
            # Check for consistent timestamp ordering (truth in temporal data)
            timestamps = [m.timestamp for m in system_data['recent_metrics'][-50:] if hasattr(m, 'timestamp')]
            if len(timestamps) > 1:
                out_of_order = sum(1 for i in range(1, len(timestamps)) if timestamps[i] < timestamps[i-1])
                if out_of_order > 0:
                    violations.append(f"Temporal data inconsistency: {out_of_order} out-of-order timestamps")
                    score -= 20.0
                    
            # Check for measurement accuracy (consistent with physical reality)
            cpu_values = [getattr(m, 'value', 0) for m in system_data['recent_metrics'][-20:] 
                         if hasattr(m, 'name') and getattr(m, 'name', '') == 'cpu_usage']
            if cpu_values:
                invalid_cpu = [v for v in cpu_values if not (0.0 <= v <= 100.0)]
                if invalid_cpu:
                    violations.append(f"CPU usage values violate physical constraints: {invalid_cpu}")
                    score -= 25.0
                    
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing Law 3 compliance: {e}")
            score = 0.0
            violations.append(f"Assessment error: {str(e)}")
            
        return {'score': score, 'violations': violations}
        
    def _assess_law4_compliance(self, system_data: Dict) -> Dict[str, Any]:
        """Assess Law 4: Energy Stewardship compliance"""
        violations = []
        score = 100.0
        
        try:
            # Check monitoring overhead
            monitoring_overhead = system_data.get('monitoring_overhead', 0.0)
            
            overhead_thresholds = {
                MonitoringLevel.MINIMAL: 1.0,
                MonitoringLevel.STANDARD: 2.0, 
                MonitoringLevel.DETAILED: 5.0,
                MonitoringLevel.DIAGNOSTIC: 10.0
            }
            
            threshold = overhead_thresholds.get(self.monitoring_level, 2.0)
            
            if monitoring_overhead > threshold:
                violations.append(f"Monitoring overhead {monitoring_overhead:.2f}% exceeds {threshold}% threshold")
                score -= 40.0
                
            # Check for efficient resource usage patterns
            current_health = system_data.get('current_health')
            if current_health:
                # CPU efficiency check
                if current_health.cpu_usage > 80.0 and len(system_data['protocol_performance']) < 5:
                    violations.append("High CPU usage with low protocol utilization suggests inefficiency")
                    score -= 20.0
                    
                # Memory efficiency check  
                if current_health.memory_usage > 70.0:
                    # Check if memory usage is justified by workload
                    total_protocol_executions = sum(p.execution_count for p in system_data['protocol_performance'].values())
                    if total_protocol_executions < 100:  # Low activity
                        violations.append("High memory usage with low activity suggests memory leaks")
                        score -= 25.0
                        
            # Check adaptive behavior (energy stewardship principle)
            if hasattr(self, 'adaptive_adjustments_count'):
                if self.adaptive_adjustments_count == 0:
                    violations.append("No adaptive adjustments made despite varying system load")
                    score -= 15.0
            else:
                self.adaptive_adjustments_count = 0
                
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing Law 4 compliance: {e}")
            score = 0.0
            violations.append(f"Assessment error: {str(e)}")
            
        return {'score': score, 'violations': violations}
        
    def _assess_law5_compliance(self, system_data: Dict) -> Dict[str, Any]:
        """Assess Law 5: Deterministic Reliability compliance"""
        violations = []
        score = 100.0
        
        try:
            # Check for consistent, predictable behavior
            
            # Protocol reliability check
            unreliable_protocols = []
            for name, perf in system_data['protocol_performance'].items():
                if perf.success_rate < 0.95:  # Less than 95% success rate
                    unreliable_protocols.append(f"{name}: {perf.success_rate:.2%}")
                    
            if unreliable_protocols:
                violations.append(f"Protocols with poor reliability: {unreliable_protocols}")
                score -= 30.0
                
            # Consistency in timing
            execution_times = []
            for perf in system_data['protocol_performance'].values():
                if perf.execution_count > 5:  # Sufficient data points
                    avg_time = perf.total_execution_time / perf.execution_count
                    time_variance = (perf.max_execution_time - perf.min_execution_time) / max(avg_time, 0.001)
                    if time_variance > 10.0:  # More than 10x variance
                        execution_times.append(f"{perf.protocol_name}: {time_variance:.1f}x variance")
                        
            if execution_times:
                violations.append(f"High execution time variance indicates non-deterministic behavior: {execution_times}")
                score -= 25.0
                
            # Check for predictable resource consumption
            recent_cpu_samples = [getattr(m, 'value', 0) for m in system_data['recent_metrics'][-10:] 
                                if hasattr(m, 'name') and getattr(m, 'name', '') == 'cpu_usage']
            if len(recent_cpu_samples) > 5:
                cpu_std = statistics.stdev(recent_cpu_samples)
                cpu_mean = statistics.mean(recent_cpu_samples)
                if cpu_mean > 0 and (cpu_std / cpu_mean) > 0.5:  # Coefficient of variation > 0.5
                    violations.append(f"High CPU usage variability suggests non-deterministic behavior: {cpu_std:.1f}% std dev")
                    score -= 20.0
                    
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing Law 5 compliance: {e}")
            score = 0.0
            violations.append(f"Assessment error: {str(e)}")
            
        return {'score': score, 'violations': violations}
        
    def _calculate_monitoring_overhead(self) -> float:
        """Calculate the computational overhead of the monitoring system"""
        try:
            current_process = psutil.Process()
            
            # Get current process CPU and memory usage
            cpu_percent = current_process.cpu_percent()
            memory_info = current_process.memory_info()
            
            # Calculate overhead as percentage of system resources
            system_cpu_count = psutil.cpu_count()
            system_memory = psutil.virtual_memory().total
            
            # Estimate monitoring overhead (simplified calculation)
            cpu_overhead = cpu_percent / system_cpu_count if system_cpu_count > 0 else 0.0
            memory_overhead = (memory_info.rss / system_memory) * 100 if system_memory > 0 else 0.0
            
            # Return combined overhead estimate
            return max(cpu_overhead, memory_overhead)
            
        except Exception as e:
            logger.error(f"Error calculating monitoring overhead: {e}")
            return 0.0
            
    def _check_compliance_violations(self, compliance_status: FiveLawsComplianceStatus):
        """Check compliance status and generate alerts for violations"""
        current_time = time.time()
        
        # Overall compliance alert
        if compliance_status.overall_score < 70.0:
            self._generate_alert(
                alert_id=f"compliance_critical_{int(current_time)}",
                severity=AlertSeverity.CRITICAL,
                title="Critical Compliance Violation",
                description=f"Overall Five Laws compliance score is {compliance_status.overall_score:.1f}%",
                source="compliance_monitor",
                tags={"score": str(compliance_status.overall_score)},
                suggested_actions=[
                    "Review specific law violations",
                    "Adjust system configuration",
                    "Consider reducing monitoring intensity"
                ],
                auto_recovery_possible=False
            )
            
        # Individual law violations
        law_scores = [
            ("Law 1 (Architectural Intelligence)", compliance_status.law1_score),
            ("Law 2 (Cognitive Governance)", compliance_status.law2_score),
            ("Law 3 (Truth Foundation)", compliance_status.law3_score),
            ("Law 4 (Energy Stewardship)", compliance_status.law4_score),
            ("Law 5 (Deterministic Reliability)", compliance_status.law5_score)
        ]
        
        for law_name, score in law_scores:
            if score < 60.0:
                law_key = law_name.split()[1].lower()
                violations = compliance_status.violations.get(law_key, [])
                
                self._generate_alert(
                    alert_id=f"law_violation_{law_key}_{int(current_time)}",
                    severity=AlertSeverity.HIGH,
                    title=f"{law_name} Violation",
                    description=f"{law_name} compliance score is {score:.1f}%",
                    source="compliance_monitor",
                    tags={"law": law_key, "score": str(score)},
                    suggested_actions=[f"Address violations: {', '.join(violations[:3])}"],
                    auto_recovery_possible=False
                )
                
    def _alert_processing_thread(self):
        """Alert processing and notification thread"""
        logger.info("Alert processing thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # Process alerts from queue
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                    self._process_alert(alert)
                    self.alert_queue.task_done()
                except queue.Empty:
                    continue
                    
                # Clean up old alerts
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in alert processing: {e}")
                
    def _process_alert(self, alert: AlertEvent):
        """Process individual alert event"""
        try:
            # Check for alert suppression
            if self._is_alert_suppressed(alert):
                return
                
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alerts_buffer.append(alert)
            
            # Attempt auto-recovery if possible
            if alert.auto_recovery_possible:
                self._attempt_auto_recovery(alert)
                
            # Send notification
            if self.alert_callback:
                try:
                    self.alert_callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
            # Log alert
            severity_level = {
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.INFO: logging.INFO
            }.get(alert.severity, logging.INFO)
            
            logger.log(severity_level, f"ALERT: {alert.title} - {alert.description}")
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {e}")
            
    def _is_alert_suppressed(self, alert: AlertEvent) -> bool:
        """Check if alert type is currently suppressed"""
        alert_type = alert.tags.get('metric', alert.source)
        current_time = time.time()
        
        suppression_end = self.alert_suppression.get(alert_type, 0)
        return current_time < suppression_end
        
    def _attempt_auto_recovery(self, alert: AlertEvent):
        """Attempt automatic recovery for recoverable alerts"""
        try:
            if 'cpu_critical' in alert.alert_id:
                # Reduce monitoring frequency
                self.effective_collection_interval *= 1.5
                self.adaptive_adjustments_count = getattr(self, 'adaptive_adjustments_count', 0) + 1
                logger.info("Auto-recovery: Reduced monitoring frequency due to high CPU")
                
            elif 'memory_critical' in alert.alert_id:
                # Force garbage collection
                gc.collect()
                # Clear old monitoring data
                self._cleanup_monitoring_data()
                logger.info("Auto-recovery: Forced GC and cleaned monitoring data")
                
            elif 'disk_critical' in alert.alert_id:
                # Clean up old data
                self._cleanup_monitoring_data(aggressive=True)
                logger.info("Auto-recovery: Cleaned up monitoring data aggressively")
                
        except Exception as e:
            logger.error(f"Error in auto-recovery for {alert.alert_id}: {e}")
            
    def _cleanup_old_alerts(self):
        """Clean up old and resolved alerts"""
        current_time = time.time()
        alert_retention_time = 3600  # 1 hour
        
        old_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if current_time - alert.timestamp > alert_retention_time
        ]
        
        for alert_id in old_alerts:
            del self.active_alerts[alert_id]
            
    def _cleanup_monitoring_data(self, aggressive: bool = False):
        """Clean up old monitoring data to free memory"""
        if aggressive:
            # More aggressive cleanup
            self.metrics_buffer.data.clear()
            self.performance_buffer.data.clear()
            
            # Keep only recent system health data
            recent_count = min(600, len(self.system_health_history))  # 10 minutes
            self.system_health_history = deque(
                list(self.system_health_history)[-recent_count:], 
                maxlen=3600
            )
            
        else:
            # Standard cleanup - keep recent data
            recent_count = min(1000, len(self.metrics_buffer.data))
            self.metrics_buffer.data = deque(
                list(self.metrics_buffer.data)[-recent_count:],
                maxlen=self.metrics_buffer.max_size
            )
            
    def _data_aggregation_thread(self):
        """Data aggregation and analysis thread"""
        logger.info("Data aggregation thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # Perform data aggregation every 60 seconds
                self.shutdown_event.wait(60.0)
                
                if self.shutdown_event.is_set():
                    break
                    
                # Aggregate performance data
                self._aggregate_performance_data()
                
                # Clean up old data based on retention policy
                self._enforce_data_retention()
                
            except Exception as e:
                logger.error(f"Error in data aggregation: {e}")
                
    def _aggregate_performance_data(self):
        """Aggregate and analyze performance data"""
        try:
            current_time = time.time()
            
            # Calculate system performance trends
            if len(self.system_health_history) > 60:  # At least 1 minute of data
                recent_health = list(self.system_health_history)[-60:]  # Last minute
                
                avg_cpu = statistics.mean([h.cpu_usage for h in recent_health])
                avg_memory = statistics.mean([h.memory_usage for h in recent_health])
                
                # Store aggregated metric
                aggregated_metric = MonitoringMetric(
                    name="system_performance_1min_avg",
                    value=avg_cpu,
                    unit="percent",
                    timestamp=current_time,
                    tags={"aggregation": "1min", "metric_type": "cpu"},
                    source="data_aggregator"
                )
                
                self.metrics_buffer.append(aggregated_metric)
                
            # Calculate protocol performance statistics
            for protocol_name, perf_data in self.protocol_performance.items():
                if perf_data.execution_count > 0:
                    avg_execution_time = perf_data.total_execution_time / perf_data.execution_count
                    
                    perf_metric = MonitoringMetric(
                        name=f"protocol_avg_execution_time",
                        value=avg_execution_time,
                        unit="seconds",
                        timestamp=current_time,
                        tags={"protocol": protocol_name},
                        source="performance_aggregator"
                    )
                    
                    self.performance_buffer.append(perf_metric)
                    
        except Exception as e:
            logger.error(f"Error aggregating performance data: {e}")
            
    def _enforce_data_retention(self):
        """Enforce data retention policies"""
        try:
            current_time = time.time()
            retention_seconds = self.data_retention_hours * 3600
            
            # Clean metrics buffer
            cutoff_time = current_time - retention_seconds
            recent_metrics = [
                m for m in self.metrics_buffer.data
                if getattr(m, 'timestamp', current_time) >= cutoff_time
            ]
            self.metrics_buffer.data = deque(recent_metrics, maxlen=self.metrics_buffer.max_size)
            
            # Clean alerts buffer
            recent_alerts = [
                a for a in self.alerts_buffer.data
                if getattr(a, 'timestamp', current_time) >= cutoff_time
            ]
            self.alerts_buffer.data = deque(recent_alerts, maxlen=self.alerts_buffer.max_size)
            
            logger.debug(f"Data retention enforced: {len(recent_metrics)} metrics, {len(recent_alerts)} alerts retained")
            
        except Exception as e:
            logger.error(f"Error enforcing data retention: {e}")
            
    def _predictive_analysis_thread(self):
        """Predictive analysis thread for failure prediction"""
        logger.info("Predictive analysis thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # Run predictive analysis every 5 minutes
                self.shutdown_event.wait(300.0)
                
                if self.shutdown_event.is_set():
                    break
                    
                # Perform predictive analysis
                self._perform_predictive_analysis()
                
            except Exception as e:
                logger.error(f"Error in predictive analysis: {e}")
                
    def _perform_predictive_analysis(self):
        """Perform predictive failure analysis"""
        try:
            if len(self.system_health_history) < 300:  # Need at least 5 minutes of data
                return
                
            # Analyze CPU usage trends
            recent_cpu = [h.cpu_usage for h in list(self.system_health_history)[-300:]]
            self._analyze_trend_prediction(recent_cpu, "cpu_usage", "percent")
            
            # Analyze memory usage trends  
            recent_memory = [h.memory_usage for h in list(self.system_health_history)[-300:]]
            self._analyze_trend_prediction(recent_memory, "memory_usage", "percent")
            
            # Analyze protocol performance trends
            self._analyze_protocol_performance_trends()
            
        except Exception as e:
            logger.error(f"Error in predictive analysis: {e}")
            
    def _analyze_trend_prediction(self, values: List[float], metric_name: str, unit: str):
        """Analyze trend and predict future values"""
        try:
            if len(values) < 30:
                return
                
            # Simple linear trend analysis
            x = list(range(len(values)))
            n = len(values)
            
            # Calculate linear regression coefficients
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            # Slope (trend)
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # Predict value in 10 minutes (600 data points ahead at 1-second intervals)
            future_x = len(values) + 600
            predicted_value = values[-1] + slope * 600
            
            # Check for concerning trends
            if metric_name == "cpu_usage" and predicted_value > 90.0 and slope > 0.1:
                self._generate_predictive_alert(
                    f"cpu_trend_prediction",
                    AlertSeverity.MEDIUM,
                    "Predicted CPU Usage Spike",
                    f"CPU usage trending upward, predicted to reach {predicted_value:.1f}% in 10 minutes",
                    {"metric": metric_name, "predicted_value": str(predicted_value), "slope": str(slope)}
                )
                
            elif metric_name == "memory_usage" and predicted_value > 90.0 and slope > 0.1:
                self._generate_predictive_alert(
                    f"memory_trend_prediction",
                    AlertSeverity.MEDIUM,
                    "Predicted Memory Usage Spike", 
                    f"Memory usage trending upward, predicted to reach {predicted_value:.1f}% in 10 minutes",
                    {"metric": metric_name, "predicted_value": str(predicted_value), "slope": str(slope)}
                )
                
        except Exception as e:
            logger.error(f"Error in trend prediction for {metric_name}: {e}")
            
    def _analyze_protocol_performance_trends(self):
        """Analyze protocol performance trends for predictive alerts"""
        try:
            for protocol_name, perf_data in self.protocol_performance.items():
                if perf_data.execution_count < 10:  # Need sufficient data
                    continue
                    
                # Check for degrading success rate
                if perf_data.success_rate < 0.9 and perf_data.error_count > 5:
                    recent_errors = list(perf_data.recent_errors)[-10:] if perf_data.recent_errors else []
                    if len(recent_errors) > 5:  # More than 5 recent errors
                        self._generate_predictive_alert(
                            f"protocol_degradation_{protocol_name}",
                            AlertSeverity.HIGH,
                            f"Protocol Performance Degradation: {protocol_name}",
                            f"Protocol {protocol_name} showing declining performance: {perf_data.success_rate:.2%} success rate",
                            {"protocol": protocol_name, "success_rate": str(perf_data.success_rate)}
                        )
                        
        except Exception as e:
            logger.error(f"Error analyzing protocol performance trends: {e}")
            
    def _generate_alert(self, alert_id: str, severity: AlertSeverity, title: str, 
                       description: str, source: str, tags: Dict[str, str],
                       suggested_actions: List[str], auto_recovery_possible: bool = False):
        """Generate and queue an alert"""
        try:
            alert = AlertEvent(
                alert_id=alert_id,
                severity=severity,
                title=title,
                description=description,
                timestamp=time.time(),
                source=source,
                tags=tags,
                suggested_actions=suggested_actions,
                auto_recovery_possible=auto_recovery_possible
            )
            
            # Add to queue for processing
            try:
                self.alert_queue.put_nowait(alert)
            except queue.Full:
                logger.warning(f"Alert queue full, dropping alert: {alert_id}")
                
        except Exception as e:
            logger.error(f"Error generating alert {alert_id}: {e}")
            
    def _generate_predictive_alert(self, alert_type: str, severity: AlertSeverity,
                                  title: str, description: str, tags: Dict[str, str]):
        """Generate a predictive alert"""
        alert_id = f"predictive_{alert_type}_{int(time.time())}"
        self._generate_alert(
            alert_id=alert_id,
            severity=severity,
            title=f"PREDICTIVE: {title}",
            description=description,
            source="predictive_analyzer",
            tags={**tags, "alert_type": "predictive"},
            suggested_actions=["Monitor trend closely", "Consider proactive measures"],
            auto_recovery_possible=False
        )
        
    def register_protocol_execution(self, protocol_name: str, execution_time: float,
                                   success: bool, memory_usage: float = 0.0, error_details: str = None):
        """Register protocol execution for performance monitoring"""
        try:
            if protocol_name not in self.protocol_performance:
                self.protocol_performance[protocol_name] = ProtocolPerformanceData(protocol_name)
                
            perf_data = self.protocol_performance[protocol_name]
            perf_data.update_execution(execution_time, success, memory_usage)
            
            if not success and error_details:
                perf_data.recent_errors.append({
                    'timestamp': time.time(),
                    'error': error_details
                })
                
            # Generate performance metric
            metric = MonitoringMetric(
                name="protocol_execution",
                value=execution_time,
                unit="seconds", 
                timestamp=time.time(),
                tags={
                    "protocol": protocol_name,
                    "success": str(success),
                    "memory_mb": str(memory_usage)
                },
                source="protocol_monitor"
            )
            
            self.performance_buffer.append(metric)
            
        except Exception as e:
            logger.error(f"Error registering protocol execution for {protocol_name}: {e}")
            
    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status and key metrics"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            return {
                'status': 'running' if self.is_running else 'stopped',
                'uptime_seconds': uptime,
                'monitoring_level': self.monitoring_level.value,
                'system_health': {
                    'cpu_usage': self.current_system_health.cpu_usage,
                    'memory_usage': self.current_system_health.memory_usage,
                    'disk_usage': self.current_system_health.disk_usage,
                    'active_threads': self.current_system_health.active_threads
                },
                'compliance_status': {
                    'overall_score': self.current_compliance.overall_score,
                    'law1_score': self.current_compliance.law1_score,
                    'law2_score': self.current_compliance.law2_score,
                    'law3_score': self.current_compliance.law3_score,
                    'law4_score': self.current_compliance.law4_score,
                    'law5_score': self.current_compliance.law5_score
                },
                'protocol_performance': {
                    name: {
                        'execution_count': perf.execution_count,
                        'success_rate': perf.success_rate,
                        'avg_execution_time': perf.total_execution_time / max(perf.execution_count, 1),
                        'error_count': perf.error_count
                    }
                    for name, perf in self.protocol_performance.items()
                },
                'active_alerts': len(self.active_alerts),
                'buffer_sizes': {
                    'metrics': len(self.metrics_buffer.data),
                    'alerts': len(self.alerts_buffer.data),
                    'performance': len(self.performance_buffer.data),
                    'system_health': len(self.system_health_history)
                },
                'monitoring_overhead': self._calculate_monitoring_overhead()
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def get_metrics(self, metric_name: Optional[str] = None, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[MonitoringMetric]:
        """Get monitoring metrics with optional filtering"""
        try:
            # Get metrics from buffer
            if start_time or end_time:
                start_time = start_time or 0.0
                end_time = end_time or time.time()
                metrics = self.metrics_buffer.get_timerange(start_time, end_time)
            else:
                metrics = list(self.metrics_buffer.data)
                
            # Filter by metric name if specified
            if metric_name:
                metrics = [m for m in metrics if getattr(m, 'name', None) == metric_name]
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return []
            
    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                  active_only: bool = False) -> List[AlertEvent]:
        """Get alerts with optional filtering"""
        try:
            if active_only:
                alerts = list(self.active_alerts.values())
            else:
                alerts = list(self.alerts_buffer.data)
                
            # Filter by severity if specified
            if severity:
                alerts = [a for a in alerts if getattr(a, 'severity', None) == severity]
                
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []
            
    def suppress_alerts(self, alert_type: str, duration_minutes: int = 30):
        """Suppress alerts of a specific type for a duration"""
        try:
            suppression_end = time.time() + (duration_minutes * 60)
            self.alert_suppression[alert_type] = suppression_end
            logger.info(f"Suppressing {alert_type} alerts for {duration_minutes} minutes")
            
        except Exception as e:
            logger.error(f"Error suppressing alerts for {alert_type}: {e}")
            
    def stop_monitoring(self):
        """Stop the real-time monitoring system"""
        if not self.is_running:
            logger.warning("Monitoring system is not running")
            return
            
        logger.info("Stopping real-time monitoring system...")
        
        # Signal shutdown
        self.shutdown_event.set()
        self.is_running = False
        
        # Wait for threads to complete
        for thread_name, thread in self.threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Thread {thread_name} did not shut down cleanly")
                
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Real-time monitoring system stopped")
        
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute monitoring protocol command
        
        Args:
            data: Command data with 'action' and parameters
            
        Returns:
            Result of the monitoring operation
        """
        try:
            action = data.get('action', 'status')
            
            if action == 'start':
                if not self.is_running:
                    self.start_monitoring()
                return {'success': True, 'message': 'Monitoring started'}
                
            elif action == 'stop':
                if self.is_running:
                    self.stop_monitoring()
                return {'success': True, 'message': 'Monitoring stopped'}
                
            elif action == 'status':
                return {'success': True, 'status': self.get_current_status()}
                
            elif action == 'metrics':
                metric_name = data.get('metric_name')
                start_time = data.get('start_time')
                end_time = data.get('end_time')
                metrics = self.get_metrics(metric_name, start_time, end_time)
                return {'success': True, 'metrics': [m._asdict() for m in metrics]}
                
            elif action == 'alerts':
                severity = data.get('severity')
                if severity and isinstance(severity, str):
                    severity = AlertSeverity(severity)
                active_only = data.get('active_only', False)
                alerts = self.get_alerts(severity, active_only)
                return {'success': True, 'alerts': [a._asdict() for a in alerts]}
                
            elif action == 'suppress_alerts':
                alert_type = data.get('alert_type')
                duration = data.get('duration_minutes', 30)
                if alert_type:
                    self.suppress_alerts(alert_type, duration)
                    return {'success': True, 'message': f'Alerts suppressed for {alert_type}'}
                else:
                    return {'success': False, 'error': 'alert_type required'}
                    
            elif action == 'register_execution':
                # Allow external protocols to register their execution
                protocol_name = data.get('protocol_name')
                execution_time = data.get('execution_time')
                success = data.get('success', True)
                memory_usage = data.get('memory_usage', 0.0)
                error_details = data.get('error_details')
                
                if protocol_name and execution_time is not None:
                    self.register_protocol_execution(
                        protocol_name, execution_time, success, memory_usage, error_details
                    )
                    return {'success': True, 'message': 'Execution registered'}
                else:
                    return {'success': False, 'error': 'protocol_name and execution_time required'}
                    
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Error executing monitoring protocol: {e}")
            return {'success': False, 'error': str(e)}

# Example usage and testing
if __name__ == '__main__':
    import signal
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create monitoring protocol instance
    def alert_handler(alert: AlertEvent):
        """Example alert handler"""
        print(f"🚨 ALERT: [{alert.severity.value.upper()}] {alert.title}")
        print(f"   Description: {alert.description}")
        print(f"   Source: {alert.source}")
        print(f"   Suggested Actions: {', '.join(alert.suggested_actions)}")
        print()
        
    monitor = RealTimeMonitorProtocol(
        monitoring_level=MonitoringLevel.STANDARD,
        alert_callback=alert_handler,
        enable_predictive_analysis=True
    )
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down monitoring system...")
        monitor.stop_monitoring()
        sys.exit(0)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start monitoring
        print("🔍 Starting Real-Time Monitoring Protocol")
        print(f"   Monitoring Level: {monitor.monitoring_level.value}")
        print(f"   Collection Interval: {monitor.effective_collection_interval}s")
        print("   Press Ctrl+C to stop\n")
        
        monitor.start_monitoring()
        
        # Simulate some protocol executions
        import random
        
        protocol_names = ["REP", "HIP", "POCP", "EEP", "VVP", "CCP", "ESL", "MTP", "SP"]
        
        for i in range(50):
            time.sleep(2.0)
            
            # Simulate protocol execution
            protocol = random.choice(protocol_names)
            execution_time = random.uniform(0.01, 0.5)
            success = random.random() > 0.05  # 95% success rate
            memory_usage = random.uniform(1.0, 10.0)
            
            monitor.register_protocol_execution(
                protocol_name=protocol,
                execution_time=execution_time, 
                success=success,
                memory_usage=memory_usage,
                error_details="Simulated error" if not success else None
            )
            
            # Print status every 10 iterations
            if (i + 1) % 10 == 0:
                status = monitor.get_current_status()
                print(f"📊 Status Update (Iteration {i + 1}):")
                print(f"   System Health: CPU {status['system_health']['cpu_usage']:.1f}%, "
                      f"Memory {status['system_health']['memory_usage']:.1f}%")
                print(f"   Compliance Score: {status['compliance_status']['overall_score']:.1f}%")
                print(f"   Active Alerts: {status['active_alerts']}")
                print(f"   Monitoring Overhead: {status['monitoring_overhead']:.2f}%")
                print()
                
        # Keep running until interrupted
        print("✅ Simulation complete. Monitoring continues...")
        while monitor.is_running:
            time.sleep(10)
            status = monitor.get_current_status()
            print(f"📈 Monitoring Status: {status['buffer_sizes']['metrics']} metrics collected")
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        monitor.stop_monitoring()
        print("Real-Time Monitoring Protocol demonstration completed.")