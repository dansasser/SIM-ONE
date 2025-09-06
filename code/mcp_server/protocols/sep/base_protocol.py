"""
Base Protocol Class for SIM-ONE Framework

Provides the foundational structure and interface that all SIM-ONE protocols
must implement to ensure consistency and interoperability.

Author: SIM-ONE Framework
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ProtocolMetadata:
    """Metadata for protocol identification and management."""
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    dependencies: List[str]
    capabilities: List[str]

class BaseProtocol(ABC):
    """
    Abstract base class for all SIM-ONE protocols.
    
    Ensures consistent interface and behavior across all protocols
    while maintaining the Five Laws of Cognitive Governance.
    """
    
    def __init__(self):
        self.protocol_name = self.__class__.__name__
        self.version = "1.0.0"
        self.initialized = False
        self.active = False
        
        # Protocol state
        self.metadata = None
        self.config = {}
        self.stats = {}
        
        # Governance tracking
        self.governance_compliance = {
            'law_1_architectural_intelligence': True,
            'law_2_cognitive_governance': True,
            'law_3_truth_foundation': True,
            'law_4_energy_stewardship': True,
            'law_5_deterministic_reliability': True
        }
        
        logger.debug(f"BaseProtocol initialized: {self.protocol_name}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the protocol.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown of the protocol."""
        pass
    
    def get_metadata(self) -> ProtocolMetadata:
        """Get protocol metadata."""
        if not self.metadata:
            self.metadata = ProtocolMetadata(
                name=self.protocol_name,
                version=self.version,
                description=f"{self.protocol_name} implementation",
                author="SIM-ONE Framework",
                created_at=datetime.now(),
                dependencies=[],
                capabilities=[]
            )
        return self.metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            'protocol_name': self.protocol_name,
            'version': self.version,
            'initialized': self.initialized,
            'active': self.active,
            'governance_compliance': self.governance_compliance,
            **self.stats
        }
    
    def validate_five_laws_compliance(self) -> Dict[str, Any]:
        """Validate compliance with the Five Laws of Cognitive Governance."""
        return {
            'compliant': all(self.governance_compliance.values()),
            'details': self.governance_compliance,
            'protocol': self.protocol_name,
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform protocol health check."""
        return {
            'status': 'healthy' if self.initialized and self.active else 'unhealthy',
            'initialized': self.initialized,
            'active': self.active,
            'protocol': self.protocol_name,
            'version': self.version,
            'timestamp': datetime.now().isoformat()
        }
    
    def set_config(self, config: Dict[str, Any]):
        """Set protocol configuration."""
        self.config = config
        logger.debug(f"Configuration updated for {self.protocol_name}")
    
    def get_config(self) -> Dict[str, Any]:
        """Get protocol configuration."""
        return self.config.copy()
    
    def log_governance_event(self, law: str, event: str, details: Optional[Dict[str, Any]] = None):
        """Log governance-related events for compliance tracking."""
        logger.info(f"Governance Event - {self.protocol_name} - {law}: {event}")
        if details:
            logger.debug(f"Event details: {details}")
    
    def update_governance_compliance(self, law: str, compliant: bool, reason: Optional[str] = None):
        """Update governance compliance status."""
        if law in self.governance_compliance:
            old_status = self.governance_compliance[law]
            self.governance_compliance[law] = compliant
            
            if old_status != compliant:
                status = "compliant" if compliant else "non-compliant"
                logger.warning(f"{self.protocol_name} - {law} is now {status}")
                if reason:
                    logger.warning(f"Reason: {reason}")
    
    async def execute_with_governance(self, operation: str, func, *args, **kwargs):
        """Execute operation with governance tracking."""
        start_time = datetime.now()
        
        try:
            # Log operation start
            self.log_governance_event("execution", f"Starting {operation}")
            
            # Execute operation
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Log successful completion
            execution_time = (datetime.now() - start_time).total_seconds()
            self.log_governance_event("execution", f"Completed {operation} in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Log error
            execution_time = (datetime.now() - start_time).total_seconds()
            self.log_governance_event("error", f"Failed {operation} after {execution_time:.3f}s: {str(e)}")
            raise
    
    def __str__(self) -> str:
        return f"{self.protocol_name} v{self.version}"
    
    def __repr__(self) -> str:
        return f"<{self.protocol_name}(version='{self.version}', initialized={self.initialized}, active={self.active})>"

