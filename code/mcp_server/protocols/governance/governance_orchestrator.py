"""
Governance Orchestrator for SIM-ONE Framework
Coordinates governance protocol execution across the entire cognitive stack
while maintaining protocol independence and scalable composition.
"""
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class GovernancePhase(Enum):
    """Phases of governance execution in cognitive workflows"""
    INITIALIZATION = "initialization"
    PRE_EXECUTION = "pre_execution"
    EXECUTION_MONITORING = "execution_monitoring"
    POST_EXECUTION = "post_execution"
    VIOLATION_RESPONSE = "violation_response"
    COMPLIANCE_REPORTING = "compliance_reporting"

@dataclass
class GovernanceContext:
    """Context information for governance execution"""
    session_id: str
    workflow_type: str
    governance_intensity: str
    protocol_stack: List[str]
    start_time: float
    phase_metrics: Dict[str, Any]
    violations: List[Dict[str, Any]]
    compliance_score: float = 0.0

@dataclass
class GovernanceViolation:
    """Represents a governance violation detected during execution"""
    law_violated: str
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any]
    timestamp: float
    suggested_remediation: Optional[str] = None

class GovernanceOrchestrator:
    """
    Orchestrates governance protocol execution across cognitive workflows.
    Provides centralized coordination while maintaining protocol independence.
    """
    
    def __init__(self):
        self.governance_sessions: Dict[str, GovernanceContext] = {}
        self.violation_handlers = {
            "low": self._handle_low_severity_violation,
            "medium": self._handle_medium_severity_violation, 
            "high": self._handle_high_severity_violation,
            "critical": self._handle_critical_violation
        }
        self.compliance_thresholds = {
            "minimum": 0.6,
            "acceptable": 0.75,
            "good": 0.85,
            "excellent": 0.95
        }
        
    @asynccontextmanager
    async def governance_session(self, session_id: str, workflow_type: str, governance_intensity: str):
        """
        Context manager for governance sessions - ensures proper initialization and cleanup
        """
        context = GovernanceContext(
            session_id=session_id,
            workflow_type=workflow_type,
            governance_intensity=governance_intensity,
            protocol_stack=[],
            start_time=time.time(),
            phase_metrics={},
            violations=[]
        )
        
        self.governance_sessions[session_id] = context
        logger.info(f"Started governance session {session_id} for {workflow_type} workflow")
        
        try:
            yield context
        finally:
            # Cleanup and final reporting
            await self._finalize_governance_session(session_id)
            if session_id in self.governance_sessions:
                del self.governance_sessions[session_id]
            logger.info(f"Finalized governance session {session_id}")
    
    async def orchestrate_governance_phase(self, 
                                         phase: GovernancePhase,
                                         session_id: str, 
                                         governance_protocols: List[str],
                                         execution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate execution of governance protocols for a specific phase
        
        Args:
            phase: The governance phase being executed
            session_id: Active governance session identifier
            governance_protocols: List of governance protocols to execute
            execution_context: Context data for protocol execution
            
        Returns:
            Aggregated governance results from all protocols
        """
        if session_id not in self.governance_sessions:
            logger.error(f"No active governance session for {session_id}")
            return {"error": f"No active governance session: {session_id}"}
        
        context = self.governance_sessions[session_id]
        phase_start_time = time.time()
        
        logger.info(f"Orchestrating governance phase: {phase.value} with {len(governance_protocols)} protocols")
        
        # Initialize phase metrics
        context.phase_metrics[phase.value] = {
            "start_time": phase_start_time,
            "protocols_executed": [],
            "violations_detected": 0,
            "compliance_scores": {}
        }
        
        # Execute governance protocols based on phase
        phase_results = {}
        
        if phase == GovernancePhase.PRE_EXECUTION:
            phase_results = await self._execute_pre_execution_governance(governance_protocols, execution_context, context)
        elif phase == GovernancePhase.EXECUTION_MONITORING:
            phase_results = await self._execute_monitoring_governance(governance_protocols, execution_context, context)
        elif phase == GovernancePhase.POST_EXECUTION:
            phase_results = await self._execute_post_execution_governance(governance_protocols, execution_context, context)
        elif phase == GovernancePhase.VIOLATION_RESPONSE:
            phase_results = await self._handle_governance_violations(execution_context, context)
        elif phase == GovernancePhase.COMPLIANCE_REPORTING:
            phase_results = await self._generate_compliance_report(execution_context, context)
        
        # Update phase metrics
        phase_duration = time.time() - phase_start_time
        context.phase_metrics[phase.value]["duration"] = phase_duration
        context.phase_metrics[phase.value]["end_time"] = time.time()
        
        logger.info(f"Completed governance phase: {phase.value} in {phase_duration:.3f}s")
        return phase_results
    
    async def _execute_pre_execution_governance(self, 
                                              protocols: List[str], 
                                              execution_context: Dict[str, Any],
                                              context: GovernanceContext) -> Dict[str, Any]:
        """Execute pre-execution governance protocols"""
        results = {"phase": "pre_execution", "protocol_results": {}, "violations": []}
        
        for protocol_name in protocols:
            if self._is_pre_execution_protocol(protocol_name):
                try:
                    # Simulate protocol execution - in real implementation, would use ProtocolManager
                    protocol_result = await self._execute_governance_protocol(protocol_name, execution_context, context)
                    results["protocol_results"][protocol_name] = protocol_result
                    
                    # Check for violations
                    violations = self._extract_violations(protocol_result, protocol_name)
                    if violations:
                        results["violations"].extend(violations)
                        context.violations.extend(violations)
                    
                    context.phase_metrics["pre_execution"]["protocols_executed"].append(protocol_name)
                    
                except Exception as e:
                    logger.error(f"Error executing pre-execution protocol {protocol_name}: {e}")
                    results["protocol_results"][protocol_name] = {"error": str(e)}
        
        return results
    
    async def _execute_monitoring_governance(self, 
                                           protocols: List[str], 
                                           execution_context: Dict[str, Any],
                                           context: GovernanceContext) -> Dict[str, Any]:
        """Execute parallel monitoring governance protocols"""
        results = {"phase": "monitoring", "protocol_results": {}, "real_time_violations": []}
        
        monitoring_protocols = [p for p in protocols if self._is_monitoring_protocol(p)]
        
        if monitoring_protocols:
            # Execute monitoring protocols in parallel
            tasks = []
            for protocol_name in monitoring_protocols:
                task = self._execute_governance_protocol(protocol_name, execution_context, context)
                tasks.append((protocol_name, task))
            
            # Gather results as they complete
            for protocol_name, task in tasks:
                try:
                    protocol_result = await task
                    results["protocol_results"][protocol_name] = protocol_result
                    
                    # Real-time violation detection
                    violations = self._extract_violations(protocol_result, protocol_name)
                    if violations:
                        results["real_time_violations"].extend(violations)
                        # Handle real-time violations immediately
                        await self._handle_real_time_violations(violations, context)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring protocol {protocol_name}: {e}")
                    results["protocol_results"][protocol_name] = {"error": str(e)}
        
        return results
    
    async def _execute_post_execution_governance(self, 
                                               protocols: List[str], 
                                               execution_context: Dict[str, Any],
                                               context: GovernanceContext) -> Dict[str, Any]:
        """Execute post-execution governance protocols"""
        results = {"phase": "post_execution", "protocol_results": {}, "compliance_assessment": {}}
        
        post_protocols = [p for p in protocols if self._is_post_execution_protocol(p)]
        
        for protocol_name in post_protocols:
            try:
                protocol_result = await self._execute_governance_protocol(protocol_name, execution_context, context)
                results["protocol_results"][protocol_name] = protocol_result
                
                # Extract compliance scores
                compliance_score = self._extract_compliance_score(protocol_result)
                if compliance_score is not None:
                    context.phase_metrics["post_execution"]["compliance_scores"][protocol_name] = compliance_score
                
                # Check for violations
                violations = self._extract_violations(protocol_result, protocol_name)
                if violations:
                    context.violations.extend(violations)
                
                context.phase_metrics["post_execution"]["protocols_executed"].append(protocol_name)
                
            except Exception as e:
                logger.error(f"Error executing post-execution protocol {protocol_name}: {e}")
                results["protocol_results"][protocol_name] = {"error": str(e)}
        
        # Calculate overall compliance
        overall_compliance = self._calculate_overall_compliance(context)
        results["compliance_assessment"]["overall_score"] = overall_compliance
        context.compliance_score = overall_compliance
        
        return results
    
    async def _execute_governance_protocol(self, 
                                         protocol_name: str, 
                                         execution_context: Dict[str, Any],
                                         context: GovernanceContext) -> Dict[str, Any]:
        """
        Execute individual governance protocol - placeholder for actual protocol execution
        In real implementation, this would use the ProtocolManager to load and execute protocols
        """
        logger.debug(f"Executing governance protocol: {protocol_name}")
        
        # Simulate protocol execution with mock results
        # In real implementation, would be:
        # protocol = self.protocol_manager.get_protocol(protocol_name)
        # result = await protocol.execute(execution_context)
        
        mock_results = {
            "ArchitecturalIntelligenceProtocol": {
                "intelligence_emergence": True,
                "coordination_efficiency": 0.85,
                "architectural_score": 0.9,
                "violations": []
            },
            "CognitiveGovernanceProtocol": {
                "governance_coverage": 0.95,
                "specialized_protocols": True,
                "quality_assurance": True,
                "governance_score": 0.92,
                "violations": []
            },
            "TruthFoundationProtocol": {
                "truth_grounding": True,
                "factual_accuracy": 0.88,
                "logical_consistency": 0.91,
                "truth_score": 0.89,
                "violations": []
            },
            "EnergyStewardshipProtocol": {
                "computational_efficiency": 0.87,
                "resource_optimization": True,
                "energy_score": 0.86,
                "violations": []
            },
            "DeterministicReliabilityProtocol": {
                "deterministic_behavior": True,
                "output_consistency": 0.94,
                "reliability_score": 0.93,
                "violations": []
            }
        }
        
        return mock_results.get(protocol_name, {"status": "protocol_not_implemented", "score": 0.5})
    
    def _is_pre_execution_protocol(self, protocol_name: str) -> bool:
        """Determine if protocol should run in pre-execution phase"""
        pre_execution_protocols = [
            "ArchitecturalIntelligenceProtocol",
            "CognitiveGovernanceProtocol",
            "ErrorEvaluationProtocol"  # For high-risk contexts
        ]
        return protocol_name in pre_execution_protocols
    
    def _is_monitoring_protocol(self, protocol_name: str) -> bool:
        """Determine if protocol should run as monitoring"""
        monitoring_protocols = [
            "ComplianceMonitoringProtocol",
            "EnergyStewardshipProtocol",
            "PerformanceMonitoringProtocol"
        ]
        return protocol_name in monitoring_protocols
    
    def _is_post_execution_protocol(self, protocol_name: str) -> bool:
        """Determine if protocol should run in post-execution phase"""
        post_execution_protocols = [
            "TruthFoundationProtocol",
            "DeterministicReliabilityProtocol",
            "ConstitutionalGovernanceProtocol",
            "ProceduralOutputControlProtocol"
        ]
        return protocol_name in post_execution_protocols
    
    def _extract_violations(self, protocol_result: Dict[str, Any], protocol_name: str) -> List[GovernanceViolation]:
        """Extract violations from protocol execution results"""
        violations = []
        
        if "violations" in protocol_result:
            for violation_data in protocol_result["violations"]:
                violation = GovernanceViolation(
                    law_violated=violation_data.get("law", "unknown"),
                    violation_type=violation_data.get("type", "compliance"),
                    severity=violation_data.get("severity", "medium"),
                    description=violation_data.get("description", f"Violation detected by {protocol_name}"),
                    context=violation_data.get("context", {}),
                    timestamp=time.time(),
                    suggested_remediation=violation_data.get("remediation")
                )
                violations.append(violation)
        
        # Check for implicit violations based on scores
        if protocol_name == "TruthFoundationProtocol":
            truth_score = protocol_result.get("truth_score", 1.0)
            if truth_score < 0.7:
                violation = GovernanceViolation(
                    law_violated="Law3_TruthFoundation",
                    violation_type="truth_grounding_insufficient",
                    severity="medium" if truth_score > 0.5 else "high",
                    description=f"Truth foundation score {truth_score} below acceptable threshold",
                    context={"score": truth_score, "threshold": 0.7},
                    timestamp=time.time(),
                    suggested_remediation="Improve factual grounding and logical consistency"
                )
                violations.append(violation)
        
        return violations
    
    def _extract_compliance_score(self, protocol_result: Dict[str, Any]) -> Optional[float]:
        """Extract compliance score from protocol result"""
        score_keys = ["score", "compliance_score", "governance_score", "truth_score", 
                     "reliability_score", "energy_score", "architectural_score"]
        
        for key in score_keys:
            if key in protocol_result:
                return protocol_result[key]
        
        return None
    
    def _calculate_overall_compliance(self, context: GovernanceContext) -> float:
        """Calculate overall compliance score for the session"""
        if "post_execution" not in context.phase_metrics:
            return 0.0
        
        compliance_scores = context.phase_metrics["post_execution"].get("compliance_scores", {})
        
        if not compliance_scores:
            return 0.0
        
        # Weight scores by importance (Five Laws get higher weight)
        weights = {
            "ArchitecturalIntelligenceProtocol": 0.2,
            "CognitiveGovernanceProtocol": 0.2,
            "TruthFoundationProtocol": 0.25,
            "EnergyStewardshipProtocol": 0.15,
            "DeterministicReliabilityProtocol": 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for protocol, score in compliance_scores.items():
            weight = weights.get(protocol, 0.1)  # Default weight for other protocols
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _handle_real_time_violations(self, violations: List[GovernanceViolation], context: GovernanceContext):
        """Handle violations detected during real-time monitoring"""
        for violation in violations:
            if violation.severity == "critical":
                logger.critical(f"CRITICAL governance violation detected: {violation.description}")
                # In production, might halt execution or trigger emergency protocols
            elif violation.severity == "high":
                logger.warning(f"HIGH severity governance violation: {violation.description}")
            
            # Store violation in context for later analysis
            context.violations.append(violation)
    
    async def _handle_governance_violations(self, execution_context: Dict[str, Any], context: GovernanceContext) -> Dict[str, Any]:
        """Handle all accumulated governance violations"""
        results = {"violations_processed": 0, "remediation_actions": []}
        
        for violation in context.violations:
            handler = self.violation_handlers.get(violation.severity, self._handle_medium_severity_violation)
            remediation = await handler(violation, execution_context, context)
            
            if remediation:
                results["remediation_actions"].append({
                    "violation": violation.description,
                    "action": remediation
                })
        
        results["violations_processed"] = len(context.violations)
        return results
    
    async def _handle_low_severity_violation(self, violation: GovernanceViolation, execution_context: Dict[str, Any], context: GovernanceContext) -> Optional[str]:
        """Handle low severity governance violations"""
        logger.info(f"Low severity violation: {violation.description}")
        return "logged_for_review"
    
    async def _handle_medium_severity_violation(self, violation: GovernanceViolation, execution_context: Dict[str, Any], context: GovernanceContext) -> Optional[str]:
        """Handle medium severity governance violations"""
        logger.warning(f"Medium severity violation: {violation.description}")
        return "flagged_for_optimization"
    
    async def _handle_high_severity_violation(self, violation: GovernanceViolation, execution_context: Dict[str, Any], context: GovernanceContext) -> Optional[str]:
        """Handle high severity governance violations"""
        logger.error(f"High severity violation: {violation.description}")
        return "requires_immediate_attention"
    
    async def _handle_critical_violation(self, violation: GovernanceViolation, execution_context: Dict[str, Any], context: GovernanceContext) -> Optional[str]:
        """Handle critical governance violations"""
        logger.critical(f"CRITICAL violation: {violation.description}")
        return "execution_halted_pending_review"
    
    async def _generate_compliance_report(self, execution_context: Dict[str, Any], context: GovernanceContext) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "session_id": context.session_id,
            "workflow_type": context.workflow_type,
            "governance_intensity": context.governance_intensity,
            "execution_duration": time.time() - context.start_time,
            "overall_compliance_score": context.compliance_score,
            "compliance_level": self._get_compliance_level(context.compliance_score),
            "violations_count": len(context.violations),
            "violations_by_severity": self._count_violations_by_severity(context.violations),
            "phase_performance": context.phase_metrics,
            "recommendations": self._generate_recommendations(context)
        }
        
        logger.info(f"Generated compliance report for session {context.session_id}: {report['compliance_level']} ({context.compliance_score:.3f})")
        return report
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level description based on score"""
        if score >= self.compliance_thresholds["excellent"]:
            return "excellent"
        elif score >= self.compliance_thresholds["good"]:
            return "good"
        elif score >= self.compliance_thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.compliance_thresholds["minimum"]:
            return "minimum"
        else:
            return "insufficient"
    
    def _count_violations_by_severity(self, violations: List[GovernanceViolation]) -> Dict[str, int]:
        """Count violations by severity level"""
        counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for violation in violations:
            counts[violation.severity] = counts.get(violation.severity, 0) + 1
        return counts
    
    def _generate_recommendations(self, context: GovernanceContext) -> List[str]:
        """Generate improvement recommendations based on governance results"""
        recommendations = []
        
        if context.compliance_score < self.compliance_thresholds["acceptable"]:
            recommendations.append("Increase governance intensity for this workflow type")
        
        violation_counts = self._count_violations_by_severity(context.violations)
        if violation_counts["critical"] > 0:
            recommendations.append("Critical violations detected - review and enhance governance protocols")
        if violation_counts["high"] > 2:
            recommendations.append("Multiple high-severity violations - consider protocol stack optimization")
        
        # Phase-specific recommendations
        if "pre_execution" in context.phase_metrics:
            pre_duration = context.phase_metrics["pre_execution"].get("duration", 0)
            if pre_duration > 2.0:  # seconds
                recommendations.append("Pre-execution governance taking too long - optimize protocol stack")
        
        return recommendations
    
    async def _finalize_governance_session(self, session_id: str):
        """Finalize governance session with cleanup and archival"""
        if session_id in self.governance_sessions:
            context = self.governance_sessions[session_id]
            
            # Generate final compliance report
            final_report = await self._generate_compliance_report({}, context)
            
            # Archive session data for future analysis (would save to database in production)
            logger.info(f"Archiving governance session {session_id} with compliance score {context.compliance_score:.3f}")
            
            # Cleanup resources
            context.violations.clear()
            context.phase_metrics.clear()


# Convenience functions for governance orchestration
async def orchestrate_full_governance_workflow(session_id: str, 
                                             workflow_type: str, 
                                             governance_protocols: List[str],
                                             execution_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to orchestrate complete governance workflow across all phases
    """
    orchestrator = GovernanceOrchestrator()
    
    async with orchestrator.governance_session(session_id, workflow_type, "standard") as context:
        results = {
            "pre_execution": await orchestrator.orchestrate_governance_phase(
                GovernancePhase.PRE_EXECUTION, session_id, governance_protocols, execution_context
            ),
            "monitoring": await orchestrator.orchestrate_governance_phase(
                GovernancePhase.EXECUTION_MONITORING, session_id, governance_protocols, execution_context
            ),
            "post_execution": await orchestrator.orchestrate_governance_phase(
                GovernancePhase.POST_EXECUTION, session_id, governance_protocols, execution_context
            ),
            "violation_response": await orchestrator.orchestrate_governance_phase(
                GovernancePhase.VIOLATION_RESPONSE, session_id, governance_protocols, execution_context
            ),
            "compliance_report": await orchestrator.orchestrate_governance_phase(
                GovernancePhase.COMPLIANCE_REPORTING, session_id, governance_protocols, execution_context
            )
        }
    
    return results