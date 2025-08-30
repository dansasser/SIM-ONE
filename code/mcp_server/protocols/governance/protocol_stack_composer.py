"""
Protocol Stack Composer for SIM-ONE Framework
Enables dynamic composition of governance protocols into existing workflows
without modifying base protocols - maintains protocol independence and scalability.
"""
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StackPosition(Enum):
    """Defines where governance protocols can be stacked in workflows"""
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"  
    PARALLEL_MONITORING = "parallel_monitoring"
    CONDITIONAL = "conditional"
    INTERCEPT = "intercept"

class GovernanceIntensity(Enum):
    """Defines intensity levels for governance enforcement"""
    MINIMAL = "minimal"      # Basic compliance checking
    STANDARD = "standard"    # Full Five Laws enforcement
    STRICT = "strict"        # Constitutional + Five Laws
    MAXIMUM = "maximum"      # All governance protocols active

@dataclass
class GovernanceRequirement:
    """Defines governance requirements for a workflow context"""
    laws_required: List[str]  # Which of the Five Laws to enforce
    intensity: GovernanceIntensity
    context_type: str        # cognitive, research, creative, etc.
    stack_positions: List[StackPosition]
    custom_protocols: Optional[List[str]] = None

class ProtocolStackComposer:
    """
    Composes governance protocols into existing workflows using stackable architecture.
    Maintains protocol independence while adding comprehensive governance.
    """
    
    def __init__(self):
        self.governance_registry = {
            # Five Laws Enforcement Protocols
            "law1": "ArchitecturalIntelligenceProtocol",
            "law2": "CognitiveGovernanceProtocol", 
            "law3": "TruthFoundationProtocol",
            "law4": "EnergyStewardshipProtocol",
            "law5": "DeterministicReliabilityProtocol",
            
            # Specialized Governance Protocols
            "constitutional": "ConstitutionalGovernanceProtocol",
            "error_evaluation": "ErrorEvaluationProtocol",
            "compliance_monitoring": "ComplianceMonitoringProtocol",
            "output_control": "ProceduralOutputControlProtocol",
            
            # Meta-Governance Protocols  
            "governance_orchestrator": "GovernanceOrchestratorProtocol",
            "stack_optimizer": "StackOptimizerProtocol"
        }
        
        self.predefined_stacks = {
            GovernanceIntensity.MINIMAL: ["law5"],  # Just deterministic reliability
            GovernanceIntensity.STANDARD: ["law1", "law2", "law3", "law4", "law5"],
            GovernanceIntensity.STRICT: ["law1", "law2", "law3", "law4", "law5", "constitutional"],
            GovernanceIntensity.MAXIMUM: list(self.governance_registry.keys())
        }
    
    def compose_governance_stack(self, 
                                base_workflow: List[Dict[str, Any]], 
                                governance_req: GovernanceRequirement) -> List[Dict[str, Any]]:
        """
        Compose governance protocols into a base workflow without modifying existing protocols.
        
        Args:
            base_workflow: Original workflow definition
            governance_req: Governance requirements specification
            
        Returns:
            Enhanced workflow with governance protocols stacked appropriately
        """
        logger.info(f"Composing governance stack with intensity: {governance_req.intensity}")
        
        enhanced_workflow = []
        
        # Get required governance protocols
        governance_protocols = self._select_governance_protocols(governance_req)
        
        # Pre-execution governance stack
        if StackPosition.PRE_EXECUTION in governance_req.stack_positions:
            pre_protocols = self._get_pre_execution_protocols(governance_protocols, governance_req.context_type)
            enhanced_workflow.extend([{"step": protocol} for protocol in pre_protocols])
        
        # Parallel monitoring stack (if specified)
        if StackPosition.PARALLEL_MONITORING in governance_req.stack_positions:
            monitoring_protocols = self._get_monitoring_protocols(governance_protocols)
            if monitoring_protocols:
                parallel_stack = {
                    "parallel": [{"step": protocol} for protocol in monitoring_protocols]
                }
                enhanced_workflow.append(parallel_stack)
        
        # Insert base workflow (unchanged)
        enhanced_workflow.extend(base_workflow)
        
        # Post-execution compliance stack
        if StackPosition.POST_EXECUTION in governance_req.stack_positions:
            post_protocols = self._get_post_execution_protocols(governance_protocols, governance_req.context_type)
            enhanced_workflow.extend([{"step": protocol} for protocol in post_protocols])
        
        # Add conditional governance if specified
        if StackPosition.CONDITIONAL in governance_req.stack_positions:
            conditional_governance = self._create_conditional_governance(governance_protocols, governance_req)
            if conditional_governance:
                enhanced_workflow.append(conditional_governance)
        
        logger.info(f"Enhanced workflow with {len(enhanced_workflow)} total steps (was {len(base_workflow)})")
        return enhanced_workflow
    
    def _select_governance_protocols(self, governance_req: GovernanceRequirement) -> List[str]:
        """Select appropriate governance protocols based on requirements"""
        
        # Start with predefined stack for the intensity level
        base_protocols = self.predefined_stacks.get(governance_req.intensity, [])
        
        # Add specific laws if requested
        if governance_req.laws_required:
            for law in governance_req.laws_required:
                law_key = f"law{law}" if law.isdigit() else law
                if law_key in self.governance_registry and law_key not in base_protocols:
                    base_protocols.append(law_key)
        
        # Add custom protocols
        if governance_req.custom_protocols:
            base_protocols.extend(governance_req.custom_protocols)
        
        # Convert keys to protocol names
        protocol_names = []
        for protocol_key in base_protocols:
            if protocol_key in self.governance_registry:
                protocol_names.append(self.governance_registry[protocol_key])
            else:
                # Assume it's already a protocol name
                protocol_names.append(protocol_key)
        
        return protocol_names
    
    def _get_pre_execution_protocols(self, governance_protocols: List[str], context_type: str) -> List[str]:
        """Get protocols that should run before main workflow execution"""
        pre_execution = []
        
        # Architectural Intelligence always runs first to validate coordination approach
        if "ArchitecturalIntelligenceProtocol" in governance_protocols:
            pre_execution.append("ArchitecturalIntelligenceProtocol")
        
        # Cognitive Governance runs early to establish governance context
        if "CognitiveGovernanceProtocol" in governance_protocols:
            pre_execution.append("CognitiveGovernanceProtocol")
        
        # Error Evaluation can run pre-emptively for certain contexts
        if context_type in ["high_risk", "production"] and "ErrorEvaluationProtocol" in governance_protocols:
            pre_execution.append("ErrorEvaluationProtocol")
        
        return pre_execution
    
    def _get_post_execution_protocols(self, governance_protocols: List[str], context_type: str) -> List[str]:
        """Get protocols that should run after main workflow execution"""
        post_execution = []
        
        # Truth Foundation validates reasoning results
        if "TruthFoundationProtocol" in governance_protocols:
            post_execution.append("TruthFoundationProtocol")
        
        # Deterministic Reliability validates output consistency  
        if "DeterministicReliabilityProtocol" in governance_protocols:
            post_execution.append("DeterministicReliabilityProtocol")
        
        # Energy Stewardship analyzes efficiency after execution
        if "EnergyStewardshipProtocol" in governance_protocols:
            post_execution.append("EnergyStewardshipProtocol")
        
        # Constitutional governance validates final outputs
        if "ConstitutionalGovernanceProtocol" in governance_protocols:
            post_execution.append("ConstitutionalGovernanceProtocol")
        
        # Output control formats final results
        if "ProceduralOutputControlProtocol" in governance_protocols:
            post_execution.append("ProceduralOutputControlProtocol")
        
        return post_execution
    
    def _get_monitoring_protocols(self, governance_protocols: List[str]) -> List[str]:
        """Get protocols that should run in parallel for monitoring"""
        monitoring = []
        
        # Compliance monitoring runs in parallel
        if "ComplianceMonitoringProtocol" in governance_protocols:
            monitoring.append("ComplianceMonitoringProtocol")
        
        # Energy monitoring can run in parallel
        if "EnergyStewardshipProtocol" in governance_protocols:
            monitoring.append("EnergyStewardshipProtocol") 
        
        return monitoring
    
    def _create_conditional_governance(self, governance_protocols: List[str], governance_req: GovernanceRequirement) -> Optional[Dict[str, Any]]:
        """Create conditional governance based on workflow context"""
        # For now, return None - conditional governance can be implemented as needed
        # This would involve creating conditional logic based on intermediate results
        return None
    
    def create_context_specific_workflow(self, 
                                       base_workflow: List[Dict[str, Any]], 
                                       context_type: str,
                                       intensity: GovernanceIntensity = GovernanceIntensity.STANDARD) -> List[Dict[str, Any]]:
        """
        Convenience method to create context-specific governance workflows
        
        Args:
            base_workflow: Original workflow
            context_type: Type of cognitive context (research, creative, production, etc.)
            intensity: Governance intensity level
        
        Returns:
            Context-optimized workflow with appropriate governance stack
        """
        
        # Define context-specific governance requirements
        context_configs = {
            "research": GovernanceRequirement(
                laws_required=["1", "3", "5"],  # Intelligence, Truth, Reliability
                intensity=intensity,
                context_type="research",
                stack_positions=[StackPosition.PRE_EXECUTION, StackPosition.POST_EXECUTION]
            ),
            "creative": GovernanceRequirement(
                laws_required=["1", "2", "4"],  # Intelligence, Governance, Efficiency  
                intensity=intensity,
                context_type="creative",
                stack_positions=[StackPosition.PRE_EXECUTION, StackPosition.PARALLEL_MONITORING]
            ),
            "production": GovernanceRequirement(
                laws_required=["1", "2", "3", "4", "5"],  # All Five Laws
                intensity=GovernanceIntensity.STRICT,
                context_type="production", 
                stack_positions=[StackPosition.PRE_EXECUTION, StackPosition.POST_EXECUTION, StackPosition.PARALLEL_MONITORING],
                custom_protocols=["constitutional", "error_evaluation"]
            ),
            "experimental": GovernanceRequirement(
                laws_required=["1", "4"],  # Intelligence, Efficiency
                intensity=GovernanceIntensity.MINIMAL,
                context_type="experimental",
                stack_positions=[StackPosition.POST_EXECUTION]
            )
        }
        
        governance_req = context_configs.get(context_type)
        if not governance_req:
            # Default to standard governance
            governance_req = GovernanceRequirement(
                laws_required=["1", "2", "3", "4", "5"],
                intensity=intensity,
                context_type="default",
                stack_positions=[StackPosition.PRE_EXECUTION, StackPosition.POST_EXECUTION]
            )
        
        return self.compose_governance_stack(base_workflow, governance_req)
    
    def optimize_stack_performance(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize governance stack for performance while maintaining compliance.
        Can merge compatible protocols or reorder for efficiency.
        """
        # For now, return workflow unchanged
        # Future optimization: merge compatible governance protocols, reorder for efficiency
        logger.info("Stack performance optimization - placeholder for future implementation")
        return workflow
    
    def validate_stack_composition(self, workflow: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate that the composed governance stack meets SIM-ONE architectural principles.
        
        Returns:
            Validation results with compliance status and recommendations
        """
        validation_results = {
            "is_valid": True,
            "compliance_score": 0.0,
            "violations": [],
            "recommendations": [],
            "five_laws_coverage": {}
        }
        
        # Check Five Laws coverage
        protocol_names = self._extract_protocol_names(workflow)
        laws_covered = []
        
        if "ArchitecturalIntelligenceProtocol" in protocol_names:
            laws_covered.append("Law1_ArchitecturalIntelligence")
        if "CognitiveGovernanceProtocol" in protocol_names:
            laws_covered.append("Law2_CognitiveGovernance") 
        if "TruthFoundationProtocol" in protocol_names:
            laws_covered.append("Law3_TruthFoundation")
        if "EnergyStewardshipProtocol" in protocol_names:
            laws_covered.append("Law4_EnergyyStewardship")
        if "DeterministicReliabilityProtocol" in protocol_names:
            laws_covered.append("Law5_DeterministicReliability")
        
        validation_results["five_laws_coverage"] = {
            "covered_laws": laws_covered,
            "coverage_percentage": len(laws_covered) / 5.0 * 100,
            "missing_laws": [f"Law{i+1}" for i in range(5) if f"Law{i+1}_" not in str(laws_covered)]
        }
        
        # Calculate compliance score
        compliance_score = len(laws_covered) / 5.0
        validation_results["compliance_score"] = compliance_score
        
        if compliance_score < 0.6:
            validation_results["violations"].append("Insufficient Five Laws coverage")
            validation_results["is_valid"] = False
        
        # Check for architectural principles
        if len(protocol_names) > 15:
            validation_results["recommendations"].append("Consider protocol stack optimization - high protocol count")
        
        return validation_results
    
    def _extract_protocol_names(self, workflow: List[Dict[str, Any]]) -> List[str]:
        """Extract all protocol names from a workflow definition"""
        protocol_names = []
        
        for step in workflow:
            if "step" in step:
                protocol_names.append(step["step"])
            elif "parallel" in step:
                for parallel_step in step["parallel"]:
                    if "step" in parallel_step:
                        protocol_names.append(parallel_step["step"])
            elif "loop" in step and "steps" in step:
                # Recursively extract from loop steps
                loop_protocols = self._extract_protocol_names(step["steps"])
                protocol_names.extend(loop_protocols)
        
        return protocol_names


# Convenience functions for common governance patterns
def create_five_laws_workflow(base_workflow: List[Dict[str, Any]], 
                             intensity: GovernanceIntensity = GovernanceIntensity.STANDARD) -> List[Dict[str, Any]]:
    """
    Convenience function to add complete Five Laws governance to any workflow
    """
    composer = ProtocolStackComposer()
    governance_req = GovernanceRequirement(
        laws_required=["1", "2", "3", "4", "5"],
        intensity=intensity,
        context_type="general",
        stack_positions=[StackPosition.PRE_EXECUTION, StackPosition.POST_EXECUTION]
    )
    return composer.compose_governance_stack(base_workflow, governance_req)

def create_production_workflow(base_workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for production-grade workflows with maximum governance
    """
    composer = ProtocolStackComposer()
    return composer.create_context_specific_workflow(base_workflow, "production")

def create_research_workflow(base_workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for research workflows with truth-focused governance
    """
    composer = ProtocolStackComposer()
    return composer.create_context_specific_workflow(base_workflow, "research")

def create_creative_workflow(base_workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function for creative workflows with balanced governance
    """
    composer = ProtocolStackComposer()
    return composer.create_context_specific_workflow(base_workflow, "creative")