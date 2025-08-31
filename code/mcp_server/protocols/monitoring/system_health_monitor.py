"""
System Health Monitor for SIM-ONE Framework

This component provides comprehensive system health monitoring capabilities,
focusing on hardware resources, system performance, and operational metrics
while adhering to the Five Laws of Cognitive Governance.

Key Features:
- Real-time hardware monitoring (CPU, memory, disk, network)
- Process and thread monitoring
- System resource utilization analysis
- Performance bottleneck detection
- Energy-efficient monitoring with adaptive sampling
- Historical trend analysis
- Predictive health assessment
- Integration with Five Laws compliance monitoring
"""

import time
import logging
import psutil
import threading
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, NamedTuple, Callable
from collections import deque, defaultdict
from enum import Enum
import json
import os
import platform
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    EXCELLENT = "excellent"    # All metrics optimal
    GOOD = "good"             # Minor issues, no action needed
    WARNING = "warning"       # Issues present, monitoring required
    CRITICAL = "critical"     # Immediate attention required
    UNKNOWN = "unknown"       # Unable to determine status

class ResourceType(Enum):
    """Types of system resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    PROCESSES = "processes"
    SYSTEM = "system"

@dataclass
class CPUMetrics:
    """CPU performance metrics"""
    usage_percent: float = 0.0
    user_time: float = 0.0
    system_time: float = 0.0
    idle_time: float = 0.0
    iowait: float = 0.0
    core_count: int = 0
    frequency_mhz: float = 0.0
    load_average: List[float] = field(default_factory=list)
    temperature_celsius: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    total_bytes: int = 0
    available_bytes: int = 0
    used_bytes: int = 0
    usage_percent: float = 0.0
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_percent: float = 0.0
    cached_bytes: int = 0
    buffers_bytes: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class DiskMetrics:
    """Disk usage and I/O metrics"""
    total_bytes: int = 0
    used_bytes: int = 0
    free_bytes: int = 0
    usage_percent: float = 0.0
    read_count: int = 0
    write_count: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_time_ms: int = 0
    write_time_ms: int = 0
    io_time_ms: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class NetworkMetrics:
    """Network interface metrics"""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    errors_in: int = 0
    errors_out: int = 0
    drops_in: int = 0
    drops_out: int = 0
    connections_established: int = 0
    connections_listening: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ProcessMetrics:
    """Process monitoring metrics"""
    total_processes: int = 0
    running_processes: int = 0
    sleeping_processes: int = 0
    zombie_processes: int = 0
    open_files: int = 0
    threads: int = 0
    memory_rss_bytes: int = 0
    memory_vms_bytes: int = 0
    cpu_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SystemMetrics:
    """Overall system metrics"""
    uptime_seconds: float = 0.0
    boot_time: float = 0.0
    users_count: int = 0
    context_switches: int = 0
    interrupts: int = 0
    soft_interrupts: int = 0
    system_calls: int = 0
    platform_info: str = ""
    timestamp: float = field(default_factory=time.time)

@dataclass
class HealthAssessment:
    """Comprehensive health assessment"""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    cpu_status: HealthStatus = HealthStatus.UNKNOWN
    memory_status: HealthStatus = HealthStatus.UNKNOWN
    disk_status: HealthStatus = HealthStatus.UNKNOWN
    network_status: HealthStatus = HealthStatus.UNKNOWN
    process_status: HealthStatus = HealthStatus.UNKNOWN
    system_status: HealthStatus = HealthStatus.UNKNOWN
    score: float = 0.0  # 0-100 overall health score
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

class SystemHealthMonitor:
    """
    Comprehensive system health monitoring component
    
    Provides detailed monitoring of system resources and performance metrics
    while maintaining energy efficiency and compliance with SIM-ONE principles.
    
    Features:
    - Multi-threaded resource monitoring
    - Adaptive sampling rates based on system load
    - Historical data analysis and trending
    - Predictive health assessment
    - Configurable thresholds and alerting
    - Energy-efficient data collection
    - Integration with compliance monitoring
    """
    
    def __init__(self,
                 collection_interval: float = 1.0,
                 enable_detailed_monitoring: bool = True,
                 max_history_size: int = 3600,
                 health_callback: Optional[Callable] = None):
        """
        Initialize System Health Monitor
        
        Args:
            collection_interval: Base interval for metric collection (seconds)
            enable_detailed_monitoring: Enable detailed process/network monitoring
            max_history_size: Maximum number of historical samples to retain
            health_callback: Optional callback for health status changes
        """
        self.collection_interval = collection_interval
        self.enable_detailed_monitoring = enable_detailed_monitoring
        self.max_history_size = max_history_size
        self.health_callback = health_callback
        
        # Current metrics
        self.current_cpu = CPUMetrics()
        self.current_memory = MemoryMetrics()
        self.current_disk = DiskMetrics()
        self.current_network = NetworkMetrics()
        self.current_process = ProcessMetrics()
        self.current_system = SystemMetrics()
        self.current_assessment = HealthAssessment()
        
        # Historical data
        self.cpu_history = deque(maxlen=max_history_size)
        self.memory_history = deque(maxlen=max_history_size)
        self.disk_history = deque(maxlen=max_history_size)
        self.network_history = deque(maxlen=max_history_size)
        self.process_history = deque(maxlen=max_history_size)
        self.system_history = deque(maxlen=max_history_size)
        self.assessment_history = deque(maxlen=max_history_size)
        
        # Health thresholds (configurable)
        self.thresholds = {
            'cpu_warning': 75.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'swap_warning': 50.0,
            'swap_critical': 80.0,
            'load_warning': 2.0,  # Per CPU core
            'load_critical': 4.0,
            'zombie_warning': 5,
            'zombie_critical': 20,
            'open_files_warning': 1000,
            'open_files_critical': 5000
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self.lock = threading.RLock()
        
        # Performance optimization
        self.last_collection_time = 0.0
        self.adaptive_interval = collection_interval
        self.collection_errors = 0
        
        # System information cache
        self._cache_system_info()
        
        logger.info("SystemHealthMonitor initialized")
        
    def _cache_system_info(self):
        """Cache static system information"""
        try:
            self.system_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture()[0],
                'hostname': platform.node(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'cpu_count_physical': psutil.cpu_count(logical=False),
                'total_memory': psutil.virtual_memory().total,
                'boot_time': psutil.boot_time()
            }
            
            # Get disk information
            self.disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    self.disk_info[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total': usage.total
                    }
                except (PermissionError, FileNotFoundError):
                    continue
                    
        except Exception as e:
            logger.error(f"Error caching system information: {e}")
            self.system_info = {}
            self.disk_info = {}
            
    def start_monitoring(self):
        """Start system health monitoring"""
        if self.is_monitoring:
            logger.warning("System health monitoring is already running")
            return
            
        self.is_monitoring = True
        self.shutdown_event.clear()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="SystemHealthMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("System health monitoring started")
        
    def stop_monitoring(self):
        """Stop system health monitoring"""
        if not self.is_monitoring:
            logger.warning("System health monitoring is not running")
            return
            
        self.is_monitoring = False
        self.shutdown_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("System health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("System health monitoring loop started")
        
        while not self.shutdown_event.is_set():
            try:
                start_time = time.time()
                
                # Collect all metrics
                self._collect_all_metrics()
                
                # Perform health assessment
                self._assess_system_health()
                
                # Update historical data
                self._update_history()
                
                # Calculate adaptive interval based on system load
                self._adjust_collection_interval()
                
                # Calculate sleep time
                collection_time = time.time() - start_time
                sleep_time = max(0.1, self.adaptive_interval - collection_time)
                
                self.last_collection_time = collection_time
                self.collection_errors = max(0, self.collection_errors - 1)  # Decay error count
                
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in system health monitoring loop: {e}")
                self.collection_errors += 1
                
                # Increase interval on repeated errors to reduce system load
                if self.collection_errors > 5:
                    self.adaptive_interval = min(self.adaptive_interval * 1.5, 30.0)
                    
                self.shutdown_event.wait(1.0)
                
    def _collect_all_metrics(self):
        """Collect all system health metrics"""
        try:
            # Collect CPU metrics
            self.current_cpu = self._collect_cpu_metrics()
            
            # Collect memory metrics
            self.current_memory = self._collect_memory_metrics()
            
            # Collect disk metrics
            self.current_disk = self._collect_disk_metrics()
            
            # Collect network metrics (if detailed monitoring enabled)
            if self.enable_detailed_monitoring:
                self.current_network = self._collect_network_metrics()
                self.current_process = self._collect_process_metrics()
                
            # Collect system metrics
            self.current_system = self._collect_system_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
    def _collect_cpu_metrics(self) -> CPUMetrics:
        """Collect CPU performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # CPU times
            cpu_times = psutil.cpu_times()
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0.0
            
            # Load average (Unix-like systems)
            load_avg = list(os.getloadavg()) if hasattr(os, 'getloadavg') else [0.0, 0.0, 0.0]
            
            # CPU temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries and 'cpu' in name.lower() or 'core' in name.lower():
                            temperature = entries[0].current
                            break
            except (AttributeError, OSError):
                pass
                
            return CPUMetrics(
                usage_percent=cpu_percent,
                user_time=cpu_times.user,
                system_time=cpu_times.system,
                idle_time=cpu_times.idle,
                iowait=getattr(cpu_times, 'iowait', 0.0),
                core_count=psutil.cpu_count(logical=True),
                frequency_mhz=frequency,
                load_average=load_avg,
                temperature_celsius=temperature,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
            return CPUMetrics()
            
    def _collect_memory_metrics(self) -> MemoryMetrics:
        """Collect memory usage metrics"""
        try:
            # Virtual memory
            vmem = psutil.virtual_memory()
            
            # Swap memory
            swap = psutil.swap_memory()
            
            return MemoryMetrics(
                total_bytes=vmem.total,
                available_bytes=vmem.available,
                used_bytes=vmem.used,
                usage_percent=vmem.percent,
                swap_total_bytes=swap.total,
                swap_used_bytes=swap.used,
                swap_percent=swap.percent,
                cached_bytes=getattr(vmem, 'cached', 0),
                buffers_bytes=getattr(vmem, 'buffers', 0),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
            return MemoryMetrics()
            
    def _collect_disk_metrics(self) -> DiskMetrics:
        """Collect disk usage and I/O metrics"""
        try:
            # Disk usage (root partition)
            disk_usage = psutil.disk_usage('/')
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            
            return DiskMetrics(
                total_bytes=disk_usage.total,
                used_bytes=disk_usage.used,
                free_bytes=disk_usage.free,
                usage_percent=(disk_usage.used / disk_usage.total) * 100 if disk_usage.total > 0 else 0.0,
                read_count=disk_io.read_count if disk_io else 0,
                write_count=disk_io.write_count if disk_io else 0,
                read_bytes=disk_io.read_bytes if disk_io else 0,
                write_bytes=disk_io.write_bytes if disk_io else 0,
                read_time_ms=disk_io.read_time if disk_io else 0,
                write_time_ms=disk_io.write_time if disk_io else 0,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
            return DiskMetrics()
            
    def _collect_network_metrics(self) -> NetworkMetrics:
        """Collect network interface metrics"""
        try:
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Network connections
            connections = psutil.net_connections()
            established = len([c for c in connections if c.status == 'ESTABLISHED'])
            listening = len([c for c in connections if c.status == 'LISTEN'])
            
            return NetworkMetrics(
                bytes_sent=net_io.bytes_sent if net_io else 0,
                bytes_recv=net_io.bytes_recv if net_io else 0,
                packets_sent=net_io.packets_sent if net_io else 0,
                packets_recv=net_io.packets_recv if net_io else 0,
                errors_in=net_io.errin if net_io else 0,
                errors_out=net_io.errout if net_io else 0,
                drops_in=net_io.dropin if net_io else 0,
                drops_out=net_io.dropout if net_io else 0,
                connections_established=established,
                connections_listening=listening,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
            return NetworkMetrics()
            
    def _collect_process_metrics(self) -> ProcessMetrics:
        """Collect process monitoring metrics"""
        try:
            # Process counts by status
            process_counts = defaultdict(int)
            total_processes = 0
            total_threads = 0
            total_open_files = 0
            total_memory_rss = 0
            total_memory_vms = 0
            current_process_cpu = 0.0
            
            current_process = psutil.Process()
            
            for proc in psutil.process_iter(['status', 'num_threads']):
                try:
                    process_counts[proc.info['status']] += 1
                    total_processes += 1
                    total_threads += proc.info['num_threads']
                    
                    # Get additional info for current process
                    if proc.pid == current_process.pid:
                        try:
                            total_open_files = len(current_process.open_files())
                            mem_info = current_process.memory_info()
                            total_memory_rss = mem_info.rss
                            total_memory_vms = mem_info.vms
                            current_process_cpu = current_process.cpu_percent()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            return ProcessMetrics(
                total_processes=total_processes,
                running_processes=process_counts.get('running', 0),
                sleeping_processes=process_counts.get('sleeping', 0),
                zombie_processes=process_counts.get('zombie', 0),
                open_files=total_open_files,
                threads=total_threads,
                memory_rss_bytes=total_memory_rss,
                memory_vms_bytes=total_memory_vms,
                cpu_percent=current_process_cpu,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return ProcessMetrics()
            
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect overall system metrics"""
        try:
            # System uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            # Users
            users = len(psutil.users())
            
            # System stats (if available)
            context_switches = 0
            interrupts = 0
            soft_interrupts = 0
            system_calls = 0
            
            try:
                cpu_stats = psutil.cpu_stats()
                context_switches = cpu_stats.ctx_switches
                interrupts = cpu_stats.interrupts
                soft_interrupts = cpu_stats.soft_interrupts
                system_calls = getattr(cpu_stats, 'syscalls', 0)
            except AttributeError:
                pass
                
            return SystemMetrics(
                uptime_seconds=uptime,
                boot_time=boot_time,
                users_count=users,
                context_switches=context_switches,
                interrupts=interrupts,
                soft_interrupts=soft_interrupts,
                system_calls=system_calls,
                platform_info=platform.platform(),
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()
            
    def _assess_system_health(self):
        """Perform comprehensive system health assessment"""
        try:
            issues = []
            recommendations = []
            
            # Assess CPU health
            cpu_status = self._assess_cpu_health(issues, recommendations)
            
            # Assess memory health
            memory_status = self._assess_memory_health(issues, recommendations)
            
            # Assess disk health
            disk_status = self._assess_disk_health(issues, recommendations)
            
            # Assess network health
            network_status = HealthStatus.GOOD
            if self.enable_detailed_monitoring:
                network_status = self._assess_network_health(issues, recommendations)
                
            # Assess process health
            process_status = HealthStatus.GOOD
            if self.enable_detailed_monitoring:
                process_status = self._assess_process_health(issues, recommendations)
                
            # Assess overall system health
            system_status = self._assess_overall_system_health(issues, recommendations)
            
            # Calculate overall status and score
            status_scores = {
                HealthStatus.EXCELLENT: 100,
                HealthStatus.GOOD: 80,
                HealthStatus.WARNING: 60,
                HealthStatus.CRITICAL: 20,
                HealthStatus.UNKNOWN: 0
            }
            
            statuses = [cpu_status, memory_status, disk_status, network_status, process_status, system_status]
            scores = [status_scores[status] for status in statuses]
            overall_score = statistics.mean(scores) if scores else 0.0
            
            # Determine overall status
            if overall_score >= 90:
                overall_status = HealthStatus.EXCELLENT
            elif overall_score >= 70:
                overall_status = HealthStatus.GOOD
            elif overall_score >= 50:
                overall_status = HealthStatus.WARNING
            elif overall_score >= 20:
                overall_status = HealthStatus.CRITICAL
            else:
                overall_status = HealthStatus.UNKNOWN
                
            # Update current assessment
            previous_status = self.current_assessment.overall_status
            
            self.current_assessment = HealthAssessment(
                overall_status=overall_status,
                cpu_status=cpu_status,
                memory_status=memory_status,
                disk_status=disk_status,
                network_status=network_status,
                process_status=process_status,
                system_status=system_status,
                score=overall_score,
                issues=issues,
                recommendations=recommendations,
                timestamp=time.time()
            )
            
            # Trigger callback if status changed
            if self.health_callback and previous_status != overall_status:
                try:
                    self.health_callback(self.current_assessment)
                except Exception as e:
                    logger.error(f"Error in health callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            
    def _assess_cpu_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess CPU health status"""
        try:
            cpu = self.current_cpu
            
            # Check CPU usage
            if cpu.usage_percent >= self.thresholds['cpu_critical']:
                issues.append(f"Critical CPU usage: {cpu.usage_percent:.1f}%")
                recommendations.append("Identify high-CPU processes and optimize or scale resources")
                return HealthStatus.CRITICAL
            elif cpu.usage_percent >= self.thresholds['cpu_warning']:
                issues.append(f"High CPU usage: {cpu.usage_percent:.1f}%")
                recommendations.append("Monitor CPU trends and consider optimization")
                return HealthStatus.WARNING
                
            # Check load average
            if len(cpu.load_average) > 0:
                load_per_core = cpu.load_average[0] / max(cpu.core_count, 1)
                if load_per_core >= self.thresholds['load_critical']:
                    issues.append(f"Critical system load: {load_per_core:.2f} per core")
                    recommendations.append("System is heavily loaded, investigate processes")
                    return HealthStatus.CRITICAL
                elif load_per_core >= self.thresholds['load_warning']:
                    issues.append(f"High system load: {load_per_core:.2f} per core") 
                    recommendations.append("System load is elevated, monitor closely")
                    return HealthStatus.WARNING
                    
            # Check temperature if available
            if cpu.temperature_celsius and cpu.temperature_celsius > 85:
                issues.append(f"High CPU temperature: {cpu.temperature_celsius:.1f}°C")
                recommendations.append("Check CPU cooling and system ventilation")
                return HealthStatus.WARNING
                
            return HealthStatus.EXCELLENT if cpu.usage_percent < 50 else HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing CPU health: {e}")
            return HealthStatus.UNKNOWN
            
    def _assess_memory_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess memory health status"""
        try:
            memory = self.current_memory
            
            # Check memory usage
            if memory.usage_percent >= self.thresholds['memory_critical']:
                issues.append(f"Critical memory usage: {memory.usage_percent:.1f}%")
                recommendations.append("Free memory immediately or risk system instability")
                return HealthStatus.CRITICAL
            elif memory.usage_percent >= self.thresholds['memory_warning']:
                issues.append(f"High memory usage: {memory.usage_percent:.1f}%")
                recommendations.append("Monitor memory usage and consider freeing unused memory")
                return HealthStatus.WARNING
                
            # Check swap usage
            if memory.swap_percent >= self.thresholds['swap_critical']:
                issues.append(f"Critical swap usage: {memory.swap_percent:.1f}%")
                recommendations.append("System is heavily swapping, add more RAM or reduce memory usage")
                return HealthStatus.CRITICAL
            elif memory.swap_percent >= self.thresholds['swap_warning']:
                issues.append(f"High swap usage: {memory.swap_percent:.1f}%")
                recommendations.append("System is swapping, monitor memory allocation")
                return HealthStatus.WARNING
                
            return HealthStatus.EXCELLENT if memory.usage_percent < 60 else HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing memory health: {e}")
            return HealthStatus.UNKNOWN
            
    def _assess_disk_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess disk health status"""
        try:
            disk = self.current_disk
            
            # Check disk usage
            if disk.usage_percent >= self.thresholds['disk_critical']:
                issues.append(f"Critical disk usage: {disk.usage_percent:.1f}%")
                recommendations.append("Free disk space immediately to prevent system issues")
                return HealthStatus.CRITICAL
            elif disk.usage_percent >= self.thresholds['disk_warning']:
                issues.append(f"High disk usage: {disk.usage_percent:.1f}%")
                recommendations.append("Clean up disk space or expand storage")
                return HealthStatus.WARNING
                
            # Check for excessive I/O (if we have historical data)
            if len(self.disk_history) > 10:
                recent_disks = list(self.disk_history)[-10:]
                if len(recent_disks) > 1:
                    io_times = [d.io_time_ms for d in recent_disks if d.io_time_ms > 0]
                    if io_times:
                        avg_io_time = statistics.mean(io_times)
                        if avg_io_time > 1000:  # More than 1 second average I/O time
                            issues.append(f"High disk I/O time: {avg_io_time:.0f}ms")
                            recommendations.append("Disk I/O is slow, check for disk issues or reduce I/O load")
                            return HealthStatus.WARNING
                            
            return HealthStatus.EXCELLENT if disk.usage_percent < 70 else HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing disk health: {e}")
            return HealthStatus.UNKNOWN
            
    def _assess_network_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess network health status"""
        try:
            network = self.current_network
            
            # Check for network errors
            total_packets = network.packets_sent + network.packets_recv
            total_errors = network.errors_in + network.errors_out
            total_drops = network.drops_in + network.drops_out
            
            if total_packets > 0:
                error_rate = (total_errors / total_packets) * 100
                drop_rate = (total_drops / total_packets) * 100
                
                if error_rate > 5.0:  # More than 5% error rate
                    issues.append(f"High network error rate: {error_rate:.2f}%")
                    recommendations.append("Investigate network connectivity issues")
                    return HealthStatus.WARNING
                    
                if drop_rate > 1.0:  # More than 1% drop rate
                    issues.append(f"High network drop rate: {drop_rate:.2f}%")
                    recommendations.append("Network congestion detected, check bandwidth")
                    return HealthStatus.WARNING
                    
            # Check connection counts
            if network.connections_established > 1000:
                issues.append(f"High number of established connections: {network.connections_established}")
                recommendations.append("Monitor network connection usage")
                return HealthStatus.WARNING
                
            return HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing network health: {e}")
            return HealthStatus.UNKNOWN
            
    def _assess_process_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess process health status"""
        try:
            process = self.current_process
            
            # Check for zombie processes
            if process.zombie_processes >= self.thresholds['zombie_critical']:
                issues.append(f"Critical number of zombie processes: {process.zombie_processes}")
                recommendations.append("Kill zombie processes immediately")
                return HealthStatus.CRITICAL
            elif process.zombie_processes >= self.thresholds['zombie_warning']:
                issues.append(f"High number of zombie processes: {process.zombie_processes}")
                recommendations.append("Clean up zombie processes")
                return HealthStatus.WARNING
                
            # Check open files
            if process.open_files >= self.thresholds['open_files_critical']:
                issues.append(f"Critical number of open files: {process.open_files}")
                recommendations.append("Close unused file handles to prevent resource exhaustion")
                return HealthStatus.CRITICAL
            elif process.open_files >= self.thresholds['open_files_warning']:
                issues.append(f"High number of open files: {process.open_files}")
                recommendations.append("Monitor file handle usage")
                return HealthStatus.WARNING
                
            return HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing process health: {e}")
            return HealthStatus.UNKNOWN
            
    def _assess_overall_system_health(self, issues: List[str], recommendations: List[str]) -> HealthStatus:
        """Assess overall system health"""
        try:
            system = self.current_system
            
            # Check system uptime (very long uptimes might indicate no reboots for updates)
            if system.uptime_seconds > 90 * 24 * 3600:  # More than 90 days
                issues.append(f"Very long uptime: {system.uptime_seconds / (24 * 3600):.1f} days")
                recommendations.append("Consider rebooting to apply system updates")
                return HealthStatus.WARNING
                
            # System is generally healthy if no other major issues
            return HealthStatus.GOOD
            
        except Exception as e:
            logger.error(f"Error assessing system health: {e}")
            return HealthStatus.UNKNOWN
            
    def _update_history(self):
        """Update historical data"""
        try:
            with self.lock:
                self.cpu_history.append(self.current_cpu)
                self.memory_history.append(self.current_memory)
                self.disk_history.append(self.current_disk)
                
                if self.enable_detailed_monitoring:
                    self.network_history.append(self.current_network)
                    self.process_history.append(self.current_process)
                    
                self.system_history.append(self.current_system)
                self.assessment_history.append(self.current_assessment)
                
        except Exception as e:
            logger.error(f"Error updating history: {e}")
            
    def _adjust_collection_interval(self):
        """Adjust collection interval based on system load (energy stewardship)"""
        try:
            # Adaptive interval based on CPU usage and system load
            cpu_usage = self.current_cpu.usage_percent
            
            if cpu_usage > 80:
                # Reduce monitoring frequency when system is heavily loaded
                self.adaptive_interval = min(self.collection_interval * 2.0, 10.0)
            elif cpu_usage < 20:
                # Increase monitoring frequency when system is lightly loaded
                self.adaptive_interval = max(self.collection_interval * 0.5, 0.5)
            else:
                # Return to normal interval
                self.adaptive_interval = self.collection_interval
                
        except Exception as e:
            logger.error(f"Error adjusting collection interval: {e}")
            
    def get_current_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            with self.lock:
                return {
                    'assessment': {
                        'overall_status': self.current_assessment.overall_status.value,
                        'score': self.current_assessment.score,
                        'cpu_status': self.current_assessment.cpu_status.value,
                        'memory_status': self.current_assessment.memory_status.value,
                        'disk_status': self.current_assessment.disk_status.value,
                        'network_status': self.current_assessment.network_status.value,
                        'process_status': self.current_assessment.process_status.value,
                        'system_status': self.current_assessment.system_status.value,
                        'issues': self.current_assessment.issues,
                        'recommendations': self.current_assessment.recommendations,
                        'timestamp': self.current_assessment.timestamp
                    },
                    'cpu': {
                        'usage_percent': self.current_cpu.usage_percent,
                        'load_average': self.current_cpu.load_average,
                        'core_count': self.current_cpu.core_count,
                        'frequency_mhz': self.current_cpu.frequency_mhz,
                        'temperature_celsius': self.current_cpu.temperature_celsius
                    },
                    'memory': {
                        'usage_percent': self.current_memory.usage_percent,
                        'total_gb': self.current_memory.total_bytes / (1024**3),
                        'available_gb': self.current_memory.available_bytes / (1024**3),
                        'swap_percent': self.current_memory.swap_percent
                    },
                    'disk': {
                        'usage_percent': self.current_disk.usage_percent,
                        'total_gb': self.current_disk.total_bytes / (1024**3),
                        'free_gb': self.current_disk.free_bytes / (1024**3)
                    },
                    'system': {
                        'uptime_hours': self.current_system.uptime_seconds / 3600,
                        'platform': self.current_system.platform_info,
                        'users': self.current_system.users_count
                    }
                }
                
        except Exception as e:
            logger.error(f"Error getting current health: {e}")
            return {'error': str(e)}
            
    def get_health_history(self, hours: float = 1.0) -> Dict[str, List]:
        """Get health history for specified time period"""
        try:
            cutoff_time = time.time() - (hours * 3600)
            
            with self.lock:
                cpu_data = [
                    {
                        'timestamp': cpu.timestamp,
                        'usage_percent': cpu.usage_percent,
                        'load_average': cpu.load_average[0] if cpu.load_average else 0.0
                    }
                    for cpu in self.cpu_history
                    if cpu.timestamp >= cutoff_time
                ]
                
                memory_data = [
                    {
                        'timestamp': mem.timestamp,
                        'usage_percent': mem.usage_percent,
                        'swap_percent': mem.swap_percent
                    }
                    for mem in self.memory_history
                    if mem.timestamp >= cutoff_time
                ]
                
                disk_data = [
                    {
                        'timestamp': disk.timestamp,
                        'usage_percent': disk.usage_percent
                    }
                    for disk in self.disk_history
                    if disk.timestamp >= cutoff_time
                ]
                
                assessment_data = [
                    {
                        'timestamp': assess.timestamp,
                        'overall_status': assess.overall_status.value,
                        'score': assess.score,
                        'issues_count': len(assess.issues)
                    }
                    for assess in self.assessment_history
                    if assess.timestamp >= cutoff_time
                ]
                
            return {
                'cpu': cpu_data,
                'memory': memory_data,
                'disk': disk_data,
                'assessments': assessment_data
            }
            
        except Exception as e:
            logger.error(f"Error getting health history: {e}")
            return {}
            
    def get_system_info(self) -> Dict[str, Any]:
        """Get cached system information"""
        return {
            'system_info': self.system_info,
            'disk_info': self.disk_info,
            'thresholds': self.thresholds,
            'monitoring_config': {
                'collection_interval': self.collection_interval,
                'adaptive_interval': self.adaptive_interval,
                'detailed_monitoring': self.enable_detailed_monitoring,
                'max_history_size': self.max_history_size,
                'is_monitoring': self.is_monitoring
            }
        }
        
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update health assessment thresholds"""
        try:
            with self.lock:
                self.thresholds.update(new_thresholds)
            logger.info(f"Updated health thresholds: {new_thresholds}")
            
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")

# Example usage
if __name__ == '__main__':
    import signal
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def health_status_changed(assessment: HealthAssessment):
        """Callback for health status changes"""
        print(f"🏥 Health Status Changed: {assessment.overall_status.value.upper()} (Score: {assessment.score:.1f})")
        if assessment.issues:
            print(f"   Issues: {', '.join(assessment.issues[:3])}")
        if assessment.recommendations:
            print(f"   Recommendations: {', '.join(assessment.recommendations[:2])}")
        print()
        
    # Create system health monitor
    monitor = SystemHealthMonitor(
        collection_interval=1.0,
        enable_detailed_monitoring=True,
        health_callback=health_status_changed
    )
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down system health monitor...")
        monitor.stop_monitoring()
        sys.exit(0)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start monitoring
        print("🏥 Starting System Health Monitor")
        print(f"   Collection Interval: {monitor.collection_interval}s")
        print(f"   Detailed Monitoring: {monitor.enable_detailed_monitoring}")
        print("   Press Ctrl+C to stop\n")
        
        monitor.start_monitoring()
        
        # Display status updates
        for i in range(60):  # Run for 1 minute
            time.sleep(5.0)  # Update every 5 seconds
            
            health = monitor.get_current_health()
            assessment = health['assessment']
            cpu = health['cpu']
            memory = health['memory']
            disk = health['disk']
            
            print(f"📊 Health Update (Iteration {i + 1}):")
            print(f"   Overall: {assessment['overall_status'].upper()} (Score: {assessment['score']:.1f})")
            print(f"   CPU: {cpu['usage_percent']:.1f}%, Memory: {memory['usage_percent']:.1f}%, Disk: {disk['usage_percent']:.1f}%")
            if assessment['issues']:
                print(f"   Issues: {len(assessment['issues'])} active")
            print()
            
        # Show history
        print("📈 Health History (Last 5 minutes):")
        history = monitor.get_health_history(hours=5.0/60.0)  # 5 minutes
        if history['assessments']:
            avg_score = statistics.mean([a['score'] for a in history['assessments']])
            print(f"   Average Health Score: {avg_score:.1f}")
            print(f"   Data Points Collected: {len(history['assessments'])}")
        
        # Keep running until interrupted
        print("✅ Monitor running continuously. Press Ctrl+C to stop...")
        while monitor.is_monitoring:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        monitor.stop_monitoring()
        print("System Health Monitor demonstration completed.")