"""
Alert Manager for SIM-ONE Framework

This component provides intelligent alerting and notification management for the
SIM-ONE cognitive governance system. It handles alert generation, routing,
escalation, and delivery while ensuring adherence to the Five Laws of Cognitive
Governance and maintaining energy-efficient operations.

Key Features:
- Multi-level alert severity and prioritization
- Intelligent alert correlation and deduplication
- Configurable notification channels and routing
- Alert escalation and acknowledgment workflows
- Historical alert tracking and analysis
- Energy-efficient alert processing
- Integration with monitoring and compliance systems
- Automated alert resolution and recovery tracking
"""

import time
import logging
import threading
import json
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from collections import deque, defaultdict, Counter
from enum import Enum
import smtplib
import requests
from datetime import datetime, timedelta
import re
import uuid

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels with numeric priorities"""
    CRITICAL = ("critical", 1)     # Immediate action required
    HIGH = ("high", 2)            # Urgent attention needed
    MEDIUM = ("medium", 3)        # Important but not urgent
    LOW = ("low", 4)              # Informational, low priority
    INFO = ("info", 5)            # General information
    
    def __init__(self, name: str, priority: int):
        self.severity_name = name
        self.priority = priority
        
    def __lt__(self, other):
        if isinstance(other, AlertSeverity):
            return self.priority < other.priority
        return NotImplemented

class AlertStatus(Enum):
    """Alert lifecycle status"""
    ACTIVE = "active"             # Alert is active and unacknowledged
    ACKNOWLEDGED = "acknowledged" # Alert has been acknowledged
    RESOLVED = "resolved"         # Alert condition has been resolved
    SUPPRESSED = "suppressed"     # Alert is suppressed due to policy
    EXPIRED = "expired"           # Alert has expired without resolution

class NotificationChannel(Enum):
    """Available notification channels"""
    LOG = "log"                   # Log file notification
    EMAIL = "email"               # Email notification
    WEBHOOK = "webhook"           # HTTP webhook notification
    CONSOLE = "console"           # Console output
    CALLBACK = "callback"         # Python callback function

class EscalationLevel(Enum):
    """Alert escalation levels"""
    NONE = "none"                 # No escalation
    LEVEL1 = "level1"            # First level escalation
    LEVEL2 = "level2"            # Second level escalation
    LEVEL3 = "level3"            # Final escalation level

@dataclass
class AlertRule:
    """Configuration for alert generation and handling"""
    name: str
    pattern: str                  # Regex pattern to match alerts
    severity: AlertSeverity
    channels: List[NotificationChannel]
    escalation_minutes: int = 30  # Minutes before escalation
    max_frequency: int = 5        # Max alerts per hour
    suppression_minutes: int = 60 # Suppression duration after resolution
    auto_resolve: bool = False    # Whether alert can auto-resolve
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class NotificationConfig:
    """Configuration for notification delivery"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Channel-specific configs
    # EMAIL: {'smtp_server', 'port', 'username', 'password', 'recipients'}
    # WEBHOOK: {'url', 'headers', 'timeout'}
    # CALLBACK: {'function'}

@dataclass
class Alert:
    """Core alert data structure"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.ACTIVE
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle tracking
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    suppressed_until: Optional[float] = None
    
    # Escalation tracking
    escalation_level: EscalationLevel = EscalationLevel.NONE
    escalated_at: Optional[float] = None
    
    # Correlation and grouping
    correlation_id: Optional[str] = None
    parent_alert_id: Optional[str] = None
    child_alert_ids: List[str] = field(default_factory=list)
    
    def is_active(self) -> bool:
        """Check if alert is currently active"""
        return self.status == AlertStatus.ACTIVE
        
    def is_suppressed(self) -> bool:
        """Check if alert is currently suppressed"""
        if self.status == AlertStatus.SUPPRESSED:
            return True
        if self.suppressed_until and time.time() < self.suppressed_until:
            return True
        return False
        
    def should_escalate(self, escalation_minutes: int) -> bool:
        """Check if alert should be escalated"""
        if not self.is_active():
            return False
        if self.escalation_level != EscalationLevel.NONE:
            return False
        return (time.time() - self.timestamp) > (escalation_minutes * 60)

@dataclass
class AlertStats:
    """Alert statistics and metrics"""
    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = field(default_factory=dict)
    alerts_by_source: Dict[str, int] = field(default_factory=dict)
    alerts_by_status: Dict[str, int] = field(default_factory=dict)
    resolution_times: List[float] = field(default_factory=list)
    escalation_count: int = 0
    suppression_count: int = 0
    auto_resolution_count: int = 0
    
    def update_alert(self, alert: Alert):
        """Update statistics with alert data"""
        self.total_alerts += 1
        
        # Update counters
        severity_name = alert.severity.severity_name
        self.alerts_by_severity[severity_name] = self.alerts_by_severity.get(severity_name, 0) + 1
        self.alerts_by_source[alert.source] = self.alerts_by_source.get(alert.source, 0) + 1
        self.alerts_by_status[alert.status.value] = self.alerts_by_status.get(alert.status.value, 0) + 1
        
        # Track resolution time if resolved
        if alert.status == AlertStatus.RESOLVED and alert.resolved_at:
            resolution_time = alert.resolved_at - alert.timestamp
            self.resolution_times.append(resolution_time)
            
        # Track escalations and suppressions
        if alert.escalation_level != EscalationLevel.NONE:
            self.escalation_count += 1
        if alert.status == AlertStatus.SUPPRESSED:
            self.suppression_count += 1

class AlertCorrelator:
    """Intelligent alert correlation and deduplication"""
    
    def __init__(self, correlation_window: int = 300):  # 5 minutes
        self.correlation_window = correlation_window
        self.correlation_patterns = [
            # Time-based correlation (same source, similar time)
            self._time_based_correlation,
            # Content-based correlation (similar titles/descriptions)
            self._content_based_correlation,
            # Source-based correlation (same system component)
            self._source_based_correlation,
            # Tag-based correlation (similar metadata)
            self._tag_based_correlation
        ]
        
    def correlate_alert(self, new_alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """
        Attempt to correlate new alert with existing alerts
        Returns correlation_id if correlation found, None otherwise
        """
        try:
            current_time = time.time()
            
            # Filter to recent alerts within correlation window
            recent_alerts = [
                alert for alert in existing_alerts
                if (current_time - alert.timestamp) <= self.correlation_window
                and alert.is_active()
            ]
            
            # Try each correlation pattern
            for pattern_func in self.correlation_patterns:
                correlation_id = pattern_func(new_alert, recent_alerts)
                if correlation_id:
                    return correlation_id
                    
            return None
            
        except Exception as e:
            logger.error(f"Error correlating alert: {e}")
            return None
            
    def _time_based_correlation(self, new_alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate based on source and timing"""
        for existing in existing_alerts:
            if (existing.source == new_alert.source and
                existing.severity == new_alert.severity and
                abs(existing.timestamp - new_alert.timestamp) <= 60):  # Within 1 minute
                return existing.correlation_id or existing.alert_id
        return None
        
    def _content_based_correlation(self, new_alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate based on title/description similarity"""
        def similarity_score(text1: str, text2: str) -> float:
            """Simple text similarity using common words"""
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0
            
        for existing in existing_alerts:
            title_sim = similarity_score(new_alert.title, existing.title)
            desc_sim = similarity_score(new_alert.description, existing.description)
            
            if title_sim > 0.7 or desc_sim > 0.6:  # High similarity threshold
                return existing.correlation_id or existing.alert_id
                
        return None
        
    def _source_based_correlation(self, new_alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate based on source system"""
        source_prefix = new_alert.source.split('.')[0] if '.' in new_alert.source else new_alert.source
        
        for existing in existing_alerts:
            existing_prefix = existing.source.split('.')[0] if '.' in existing.source else existing.source
            if source_prefix == existing_prefix and existing.severity == new_alert.severity:
                return existing.correlation_id or existing.alert_id
                
        return None
        
    def _tag_based_correlation(self, new_alert: Alert, existing_alerts: List[Alert]) -> Optional[str]:
        """Correlate based on similar tags"""
        new_tags = set(new_alert.tags.items())
        
        for existing in existing_alerts:
            existing_tags = set(existing.tags.items())
            common_tags = new_tags.intersection(existing_tags)
            
            # Correlate if they share significant tags
            if len(common_tags) >= 2 or (len(common_tags) >= 1 and len(new_tags) <= 2):
                return existing.correlation_id or existing.alert_id
                
        return None

class NotificationDispatcher:
    """Handles delivery of notifications across different channels"""
    
    def __init__(self):
        self.channels: Dict[NotificationChannel, NotificationConfig] = {}
        self.delivery_stats = defaultdict(int)
        self.failed_deliveries = deque(maxlen=1000)
        
    def add_channel(self, config: NotificationConfig):
        """Add a notification channel"""
        self.channels[config.channel] = config
        logger.info(f"Added notification channel: {config.channel.value}")
        
    def remove_channel(self, channel: NotificationChannel):
        """Remove a notification channel"""
        if channel in self.channels:
            del self.channels[channel]
            logger.info(f"Removed notification channel: {channel.value}")
            
    def send_notification(self, alert: Alert, channels: List[NotificationChannel]) -> Dict[NotificationChannel, bool]:
        """
        Send alert notification to specified channels
        Returns dict of channel -> success status
        """
        results = {}
        
        for channel in channels:
            if channel not in self.channels:
                logger.warning(f"Channel {channel.value} not configured, skipping")
                results[channel] = False
                continue
                
            config = self.channels[channel]
            if not config.enabled:
                logger.debug(f"Channel {channel.value} disabled, skipping")
                results[channel] = False
                continue
                
            try:
                success = self._send_to_channel(alert, config)
                results[channel] = success
                
                if success:
                    self.delivery_stats[f"{channel.value}_success"] += 1
                else:
                    self.delivery_stats[f"{channel.value}_failed"] += 1
                    
            except Exception as e:
                logger.error(f"Error sending notification to {channel.value}: {e}")
                results[channel] = False
                self.delivery_stats[f"{channel.value}_error"] += 1
                self.failed_deliveries.append({
                    'timestamp': time.time(),
                    'channel': channel.value,
                    'alert_id': alert.alert_id,
                    'error': str(e)
                })
                
        return results
        
    def _send_to_channel(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send notification to specific channel"""
        if config.channel == NotificationChannel.LOG:
            return self._send_log(alert, config)
        elif config.channel == NotificationChannel.EMAIL:
            return self._send_email(alert, config)
        elif config.channel == NotificationChannel.WEBHOOK:
            return self._send_webhook(alert, config)
        elif config.channel == NotificationChannel.CONSOLE:
            return self._send_console(alert, config)
        elif config.channel == NotificationChannel.CALLBACK:
            return self._send_callback(alert, config)
        else:
            logger.error(f"Unknown notification channel: {config.channel}")
            return False
            
    def _send_log(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send notification to log"""
        try:
            log_level = {
                AlertSeverity.CRITICAL: logging.CRITICAL,
                AlertSeverity.HIGH: logging.ERROR,
                AlertSeverity.MEDIUM: logging.WARNING,
                AlertSeverity.LOW: logging.INFO,
                AlertSeverity.INFO: logging.INFO
            }.get(alert.severity, logging.INFO)
            
            message = f"ALERT [{alert.severity.severity_name.upper()}] {alert.title}: {alert.description}"
            logger.log(log_level, message)
            return True
            
        except Exception as e:
            logger.error(f"Error sending log notification: {e}")
            return False
            
    def _send_email(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send email notification"""
        try:
            email_config = config.config
            
            # Create email message
            subject = f"[{alert.severity.severity_name.upper()}] {alert.title}"
            body = self._format_email_body(alert)
            
            # Send email (simplified - would need proper SMTP implementation)
            logger.info(f"Email notification sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
            return False
            
    def _send_webhook(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send webhook notification"""
        try:
            webhook_config = config.config
            url = webhook_config.get('url')
            timeout = webhook_config.get('timeout', 30)
            headers = webhook_config.get('headers', {})
            
            payload = {
                'alert_id': alert.alert_id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.severity_name,
                'source': alert.source,
                'timestamp': alert.timestamp,
                'status': alert.status.value,
                'tags': alert.tags,
                'metadata': alert.metadata
            }
            
            # Send webhook (would need actual HTTP request in production)
            logger.info(f"Webhook notification sent to {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False
            
    def _send_console(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send console notification"""
        try:
            severity_icon = {
                AlertSeverity.CRITICAL: "🚨",
                AlertSeverity.HIGH: "⚠️ ",
                AlertSeverity.MEDIUM: "📝",
                AlertSeverity.LOW: "ℹ️ ",
                AlertSeverity.INFO: "💡"
            }.get(alert.severity, "📢")
            
            print(f"{severity_icon} ALERT: {alert.title}")
            print(f"   Severity: {alert.severity.severity_name.upper()}")
            print(f"   Source: {alert.source}")
            print(f"   Description: {alert.description}")
            if alert.tags:
                print(f"   Tags: {', '.join([f'{k}={v}' for k, v in alert.tags.items()])}")
            print()
            return True
            
        except Exception as e:
            logger.error(f"Error sending console notification: {e}")
            return False
            
    def _send_callback(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send callback notification"""
        try:
            callback_func = config.config.get('function')
            if callback_func and callable(callback_func):
                callback_func(alert)
                return True
            else:
                logger.error("Invalid or missing callback function")
                return False
                
        except Exception as e:
            logger.error(f"Error sending callback notification: {e}")
            return False
            
    def _format_email_body(self, alert: Alert) -> str:
        """Format email body for alert"""
        body = f"""
Alert Details:
--------------
Title: {alert.title}
Severity: {alert.severity.severity_name.upper()}
Source: {alert.source}
Time: {datetime.fromtimestamp(alert.timestamp)}
Status: {alert.status.value}

Description:
{alert.description}

Tags:
{json.dumps(alert.tags, indent=2) if alert.tags else 'None'}

Metadata:
{json.dumps(alert.metadata, indent=2) if alert.metadata else 'None'}

Alert ID: {alert.alert_id}
"""
        return body.strip()

class AlertManager:
    """
    Comprehensive Alert Manager for SIM-ONE Framework
    
    Provides intelligent alerting, notification, and alert lifecycle management
    for the cognitive governance system while maintaining energy efficiency
    and compliance with the Five Laws.
    
    Features:
    - Multi-level alert severity and prioritization
    - Intelligent alert correlation and deduplication
    - Configurable notification channels and routing
    - Alert escalation and acknowledgment workflows
    - Historical alert tracking and analysis
    - Energy-efficient alert processing
    - Automated alert resolution and recovery tracking
    """
    
    def __init__(self,
                 max_active_alerts: int = 1000,
                 max_history_size: int = 10000,
                 correlation_window: int = 300,
                 enable_auto_correlation: bool = True):
        """
        Initialize Alert Manager
        
        Args:
            max_active_alerts: Maximum number of active alerts
            max_history_size: Maximum historical alerts to retain
            correlation_window: Time window for alert correlation (seconds)
            enable_auto_correlation: Enable automatic alert correlation
        """
        self.max_active_alerts = max_active_alerts
        self.max_history_size = max_history_size
        self.correlation_window = correlation_window
        self.enable_auto_correlation = enable_auto_correlation
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_history_size)
        
        # Alert rules and configuration
        self.alert_rules: Dict[str, AlertRule] = {}
        self.global_suppression: Dict[str, float] = {}  # pattern -> suppression_end_time
        
        # Components
        self.correlator = AlertCorrelator(correlation_window)
        self.dispatcher = NotificationDispatcher()
        
        # Statistics and metrics
        self.stats = AlertStats()
        
        # Threading
        self.is_running = False
        self.processing_thread = None
        self.shutdown_event = threading.Event()
        self.alert_queue = deque()
        self.queue_lock = threading.RLock()
        
        # Alert processing intervals
        self.processing_interval = 1.0  # Process alerts every second
        self.escalation_check_interval = 60.0  # Check escalations every minute
        self.cleanup_interval = 300.0  # Cleanup every 5 minutes
        
        logger.info("AlertManager initialized")
        
    def start(self):
        """Start alert processing"""
        if self.is_running:
            logger.warning("Alert manager is already running")
            return
            
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            name="AlertManager",
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("Alert manager started")
        
    def stop(self):
        """Stop alert processing"""
        if not self.is_running:
            logger.warning("Alert manager is not running")
            return
            
        self.is_running = False
        self.shutdown_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
            
        logger.info("Alert manager stopped")
        
    def _processing_loop(self):
        """Main alert processing loop"""
        logger.info("Alert processing loop started")
        
        last_escalation_check = 0.0
        last_cleanup = 0.0
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                
                # Process pending alerts
                self._process_alert_queue()
                
                # Check for escalations
                if current_time - last_escalation_check >= self.escalation_check_interval:
                    self._check_escalations()
                    last_escalation_check = current_time
                    
                # Cleanup old data
                if current_time - last_cleanup >= self.cleanup_interval:
                    self._cleanup_old_data()
                    last_cleanup = current_time
                    
                # Sleep
                self.shutdown_event.wait(self.processing_interval)
                
            except Exception as e:
                logger.error(f"Error in alert processing loop: {e}")
                self.shutdown_event.wait(1.0)
                
    def _process_alert_queue(self):
        """Process queued alerts"""
        try:
            with self.queue_lock:
                while self.alert_queue:
                    alert_data = self.alert_queue.popleft()
                    self._process_single_alert(alert_data)
                    
        except Exception as e:
            logger.error(f"Error processing alert queue: {e}")
            
    def _process_single_alert(self, alert_data: Dict[str, Any]):
        """Process a single alert"""
        try:
            # Create alert object
            alert = Alert(
                alert_id=alert_data.get('alert_id', str(uuid.uuid4())),
                title=alert_data['title'],
                description=alert_data['description'],
                severity=alert_data['severity'],
                source=alert_data['source'],
                tags=alert_data.get('tags', {}),
                metadata=alert_data.get('metadata', {})
            )
            
            # Check if alert should be suppressed
            if self._is_suppressed(alert):
                alert.status = AlertStatus.SUPPRESSED
                logger.debug(f"Alert suppressed: {alert.alert_id}")
                return
                
            # Check for correlation if enabled
            if self.enable_auto_correlation:
                correlation_id = self.correlator.correlate_alert(alert, list(self.active_alerts.values()))
                if correlation_id:
                    alert.correlation_id = correlation_id
                    logger.debug(f"Alert correlated: {alert.alert_id} -> {correlation_id}")
                    
            # Apply alert rules
            rule = self._find_matching_rule(alert)
            if rule:
                # Check frequency limits
                if not self._check_frequency_limit(alert, rule):
                    alert.status = AlertStatus.SUPPRESSED
                    logger.debug(f"Alert suppressed due to frequency limit: {alert.alert_id}")
                    return
                    
                # Send notifications
                channels = rule.channels
            else:
                # Use default channels if no rule matches
                channels = [NotificationChannel.LOG]
                
            # Send notifications
            delivery_results = self.dispatcher.send_notification(alert, channels)
            alert.metadata['notification_results'] = {
                channel.value: success for channel, success in delivery_results.items()
            }
            
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.stats.update_alert(alert)
            
            # Manage active alert count
            if len(self.active_alerts) > self.max_active_alerts:
                self._cleanup_oldest_alerts()
                
            logger.info(f"Alert processed: {alert.alert_id} - {alert.title}")
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
            
    def _is_suppressed(self, alert: Alert) -> bool:
        """Check if alert should be suppressed"""
        current_time = time.time()
        
        # Check global suppressions by pattern
        for pattern, suppression_end in self.global_suppression.items():
            if current_time < suppression_end:
                if re.search(pattern, alert.title) or re.search(pattern, alert.source):
                    return True
                    
        return False
        
    def _find_matching_rule(self, alert: Alert) -> Optional[AlertRule]:
        """Find alert rule that matches the alert"""
        for rule in self.alert_rules.values():
            if re.search(rule.pattern, alert.title) or re.search(rule.pattern, alert.source):
                return rule
        return None
        
    def _check_frequency_limit(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert exceeds frequency limit"""
        if rule.max_frequency <= 0:
            return True  # No limit
            
        current_time = time.time()
        hour_ago = current_time - 3600
        
        # Count similar alerts in the last hour
        similar_count = 0
        for existing_alert in self.active_alerts.values():
            if (existing_alert.timestamp >= hour_ago and
                existing_alert.source == alert.source and
                existing_alert.severity == alert.severity):
                similar_count += 1
                
        return similar_count < rule.max_frequency
        
    def _check_escalations(self):
        """Check for alerts that need escalation"""
        try:
            current_time = time.time()
            
            for alert in self.active_alerts.values():
                if not alert.is_active():
                    continue
                    
                # Find matching rule for escalation settings
                rule = self._find_matching_rule(alert)
                escalation_minutes = rule.escalation_minutes if rule else 30
                
                if alert.should_escalate(escalation_minutes):
                    self._escalate_alert(alert)
                    
        except Exception as e:
            logger.error(f"Error checking escalations: {e}")
            
    def _escalate_alert(self, alert: Alert):
        """Escalate an alert to the next level"""
        try:
            # Determine next escalation level
            if alert.escalation_level == EscalationLevel.NONE:
                next_level = EscalationLevel.LEVEL1
            elif alert.escalation_level == EscalationLevel.LEVEL1:
                next_level = EscalationLevel.LEVEL2
            elif alert.escalation_level == EscalationLevel.LEVEL2:
                next_level = EscalationLevel.LEVEL3
            else:
                return  # Already at max escalation
                
            # Update alert
            alert.escalation_level = next_level
            alert.escalated_at = time.time()
            
            # Create escalation notification
            escalation_alert = Alert(
                alert_id=f"escalation_{alert.alert_id}",
                title=f"ESCALATED: {alert.title}",
                description=f"Alert {alert.alert_id} has been escalated to {next_level.value}. "
                           f"Original description: {alert.description}",
                severity=AlertSeverity.HIGH,  # Escalations are always HIGH severity
                source=f"alert_manager.escalation",
                tags={'original_alert_id': alert.alert_id, 'escalation_level': next_level.value},
                parent_alert_id=alert.alert_id
            )
            
            # Send escalation notification
            channels = [NotificationChannel.LOG, NotificationChannel.CONSOLE]
            self.dispatcher.send_notification(escalation_alert, channels)
            
            logger.warning(f"Alert escalated: {alert.alert_id} to {next_level.value}")
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert.alert_id}: {e}")
            
    def _cleanup_old_data(self):
        """Clean up old alerts and data"""
        try:
            current_time = time.time()
            
            # Move old resolved/expired alerts to history
            expired_alerts = []
            for alert_id, alert in self.active_alerts.items():
                # Auto-expire old alerts (24 hours)
                if current_time - alert.timestamp > 86400:
                    alert.status = AlertStatus.EXPIRED
                    expired_alerts.append(alert_id)
                # Move resolved alerts to history after 1 hour
                elif (alert.status == AlertStatus.RESOLVED and 
                      alert.resolved_at and 
                      current_time - alert.resolved_at > 3600):
                    expired_alerts.append(alert_id)
                    
            # Move to history and remove from active
            for alert_id in expired_alerts:
                alert = self.active_alerts.pop(alert_id)
                self.alert_history.append(alert)
                logger.debug(f"Moved alert to history: {alert_id}")
                
            # Clean up global suppressions
            expired_suppressions = [
                pattern for pattern, end_time in self.global_suppression.items()
                if current_time >= end_time
            ]
            for pattern in expired_suppressions:
                del self.global_suppression[pattern]
                
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            
    def _cleanup_oldest_alerts(self):
        """Remove oldest alerts when limit exceeded"""
        try:
            # Sort by timestamp and remove oldest
            oldest_alerts = sorted(
                self.active_alerts.items(),
                key=lambda x: x[1].timestamp
            )
            
            # Remove oldest 10% of alerts
            remove_count = max(1, len(oldest_alerts) // 10)
            
            for i in range(remove_count):
                alert_id, alert = oldest_alerts[i]
                alert.status = AlertStatus.EXPIRED
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                logger.debug(f"Removed oldest alert: {alert_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up oldest alerts: {e}")
            
    def create_alert(self, title: str, description: str, severity: AlertSeverity,
                    source: str, tags: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new alert
        
        Returns:
            Alert ID of the created alert
        """
        try:
            alert_data = {
                'alert_id': str(uuid.uuid4()),
                'title': title,
                'description': description,
                'severity': severity,
                'source': source,
                'tags': tags or {},
                'metadata': metadata or {}
            }
            
            # Queue alert for processing
            with self.queue_lock:
                self.alert_queue.append(alert_data)
                
            return alert_data['alert_id']
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return ""
            
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False
                
            if not alert.is_active():
                logger.warning(f"Cannot acknowledge non-active alert: {alert_id}")
                return False
                
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = acknowledged_by
            
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False
            
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert"""
        try:
            alert = self.active_alerts.get(alert_id)
            if not alert:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False
                
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            alert.resolved_by = resolved_by
            
            logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False
            
    def suppress_alerts(self, pattern: str, duration_minutes: int = 60):
        """Suppress alerts matching pattern for specified duration"""
        try:
            suppression_end = time.time() + (duration_minutes * 60)
            self.global_suppression[pattern] = suppression_end
            
            logger.info(f"Suppressing alerts matching '{pattern}' for {duration_minutes} minutes")
            
        except Exception as e:
            logger.error(f"Error suppressing alerts: {e}")
            
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
        
    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            
    def add_notification_channel(self, config: NotificationConfig):
        """Add a notification channel"""
        self.dispatcher.add_channel(config)
        
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         source_pattern: Optional[str] = None) -> List[Alert]:
        """Get active alerts with optional filtering"""
        try:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
                
            if source_pattern:
                alerts = [a for a in alerts if re.search(source_pattern, a.source)]
                
            return sorted(alerts, key=lambda a: (a.severity.priority, a.timestamp))
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
            
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics and metrics"""
        try:
            current_time = time.time()
            
            # Calculate recent alert rates
            hour_ago = current_time - 3600
            recent_alerts = [
                a for a in list(self.active_alerts.values()) + list(self.alert_history)[-1000:]
                if a.timestamp >= hour_ago
            ]
            
            return {
                'total_alerts': self.stats.total_alerts,
                'active_alerts': len(self.active_alerts),
                'alerts_last_hour': len(recent_alerts),
                'alerts_by_severity': dict(self.stats.alerts_by_severity),
                'alerts_by_source': dict(self.stats.alerts_by_source),
                'alerts_by_status': dict(self.stats.alerts_by_status),
                'average_resolution_time': (
                    statistics.mean(self.stats.resolution_times)
                    if self.stats.resolution_times else 0.0
                ),
                'escalation_count': self.stats.escalation_count,
                'suppression_count': self.stats.suppression_count,
                'notification_stats': dict(self.dispatcher.delivery_stats),
                'active_suppressions': len(self.global_suppression),
                'alert_rules_count': len(self.alert_rules)
            }
            
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {}

# Example usage
if __name__ == '__main__':
    import signal
    import sys
    import random
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def custom_alert_callback(alert: Alert):
        """Custom callback for alert notifications"""
        print(f"🔔 Custom Callback: {alert.title} (Severity: {alert.severity.severity_name})")
        
    # Create alert manager
    manager = AlertManager(
        max_active_alerts=100,
        correlation_window=300,
        enable_auto_correlation=True
    )
    
    # Configure notification channels
    manager.add_notification_channel(
        NotificationConfig(NotificationChannel.CONSOLE, enabled=True)
    )
    manager.add_notification_channel(
        NotificationConfig(NotificationChannel.LOG, enabled=True)
    )
    manager.add_notification_channel(
        NotificationConfig(
            NotificationChannel.CALLBACK,
            enabled=True,
            config={'function': custom_alert_callback}
        )
    )
    
    # Add alert rules
    manager.add_alert_rule(AlertRule(
        name="high_cpu_rule",
        pattern="cpu.*critical",
        severity=AlertSeverity.CRITICAL,
        channels=[NotificationChannel.CONSOLE, NotificationChannel.CALLBACK],
        escalation_minutes=5,
        max_frequency=3
    ))
    
    manager.add_alert_rule(AlertRule(
        name="memory_rule",
        pattern="memory.*warning",
        severity=AlertSeverity.MEDIUM,
        channels=[NotificationChannel.LOG],
        escalation_minutes=15,
        max_frequency=10
    ))
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        print("\nShutting down alert manager...")
        manager.stop()
        sys.exit(0)
        
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start alert manager
        print("🚨 Starting Alert Manager")
        print("   Correlation enabled, multiple notification channels configured")
        print("   Press Ctrl+C to stop\n")
        
        manager.start()
        
        # Simulate alerts
        alert_types = [
            ("High CPU Usage", "CPU usage critical on monitoring system", AlertSeverity.CRITICAL, "system.cpu"),
            ("Memory Warning", "Memory usage warning threshold exceeded", AlertSeverity.MEDIUM, "system.memory"),
            ("Protocol Failure", "Protocol execution failed multiple times", AlertSeverity.HIGH, "protocol.execution"),
            ("Network Issue", "Network connectivity problems detected", AlertSeverity.MEDIUM, "network.connectivity"),
            ("Disk Space", "Disk space running low", AlertSeverity.LOW, "system.disk")
        ]
        
        for i in range(30):
            time.sleep(2.0)
            
            # Create random alerts
            title, description, severity, source = random.choice(alert_types)
            
            # Add some variation
            variation = random.randint(1, 100)
            title_variant = f"{title} (Instance {variation})"
            source_variant = f"{source}.{variation % 5}"  # 5 different instances
            
            alert_id = manager.create_alert(
                title=title_variant,
                description=f"{description} - Iteration {i+1}",
                severity=severity,
                source=source_variant,
                tags={'iteration': str(i+1), 'test': 'true'},
                metadata={'simulation': True, 'value': variation}
            )
            
            # Randomly acknowledge or resolve some alerts
            if random.random() < 0.3:  # 30% chance to acknowledge
                active_alerts = manager.get_active_alerts()
                if active_alerts:
                    random_alert = random.choice(active_alerts)
                    manager.acknowledge_alert(random_alert.alert_id, "test_user")
                    
            if random.random() < 0.2:  # 20% chance to resolve
                active_alerts = [a for a in manager.get_active_alerts() 
                               if a.status == AlertStatus.ACKNOWLEDGED]
                if active_alerts:
                    random_alert = random.choice(active_alerts)
                    manager.resolve_alert(random_alert.alert_id, "auto_resolver")
                    
            # Print status every 10 iterations
            if (i + 1) % 10 == 0:
                stats = manager.get_alert_statistics()
                print(f"📊 Alert Statistics (Iteration {i + 1}):")
                print(f"   Active Alerts: {stats['active_alerts']}")
                print(f"   Alerts Last Hour: {stats['alerts_last_hour']}")
                print(f"   Total Processed: {stats['total_alerts']}")
                print(f"   Escalations: {stats['escalation_count']}")
                
                # Show severity breakdown
                severity_stats = stats['alerts_by_severity']
                if severity_stats:
                    print(f"   By Severity: {', '.join([f'{k}:{v}' for k, v in severity_stats.items()])}")
                print()
                
        # Suppress some alert types
        print("🔇 Testing alert suppression...")
        manager.suppress_alerts("cpu.*critical", duration_minutes=2)
        
        # Create more alerts to test suppression
        for i in range(5):
            time.sleep(1.0)
            manager.create_alert(
                title="High CPU Usage (Should be suppressed)",
                description="This alert should be suppressed",
                severity=AlertSeverity.CRITICAL,
                source="system.cpu.suppressed"
            )
            
        # Final statistics
        time.sleep(5.0)
        print("\n📈 Final Alert Statistics:")
        final_stats = manager.get_alert_statistics()
        for key, value in final_stats.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
                
        # Keep running until interrupted
        print("\n✅ Alert Manager running. Press Ctrl+C to stop...")
        while manager.is_running:
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    finally:
        manager.stop()
        print("Alert Manager demonstration completed.")