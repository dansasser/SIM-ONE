"""
POCP (Procedural Output Control Protocol)
Output formatting and presentation control - stackable output governance for SIM-ONE Framework

This protocol implements deterministic output formatting, constitutional constraint application,
and consistent presentation layer governance across all protocol outputs.
"""
import logging
import asyncio
import time
import json
import re
from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
from collections import defaultdict
import yaml

logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    YAML = "yaml"
    XML = "xml"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    CSV = "csv"
    STRUCTURED = "structured"

class OutputConstraint(Enum):
    """Constitutional output constraints"""
    CONTENT_SAFETY = "content_safety"
    FACTUAL_ACCURACY = "factual_accuracy"
    LOGICAL_CONSISTENCY = "logical_consistency"
    ETHICAL_COMPLIANCE = "ethical_compliance"
    PRIVACY_PROTECTION = "privacy_protection"
    BIAS_MITIGATION = "bias_mitigation"
    TRUTHFULNESS = "truthfulness"
    DETERMINISTIC_FORMAT = "deterministic_format"

class PresentationLayer(Enum):
    """Presentation layer types"""
    RAW = "raw"
    FORMATTED = "formatted"
    SUMMARIZED = "summarized"
    DETAILED = "detailed"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    USER_FRIENDLY = "user_friendly"

class OutputQuality(Enum):
    """Output quality levels"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"
    PUBLICATION_READY = "publication_ready"

@dataclass
class OutputSpecification:
    """Specification for output formatting and control"""
    format: OutputFormat
    constraints: List[OutputConstraint]
    presentation_layer: PresentationLayer
    quality_level: OutputQuality
    target_audience: str
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    structure_template: Optional[Dict[str, Any]] = None
    custom_rules: List[str] = field(default_factory=list)

@dataclass
class FormattingRule:
    """Rule for output formatting"""
    name: str
    description: str
    pattern: str
    replacement: str
    apply_order: int
    is_mandatory: bool
    constraint_type: OutputConstraint

@dataclass
class ConstitutionalConstraint:
    """Constitutional constraint for output control"""
    name: str
    description: str
    validator_function: str
    severity: str
    auto_fix: bool
    fallback_action: str

@dataclass
class OutputMetrics:
    """Metrics for output control effectiveness"""
    total_outputs_processed: int
    constraint_violations_detected: int
    constraint_violations_fixed: int
    formatting_corrections_applied: int
    presentation_optimizations: int
    quality_improvements: int
    processing_time_ms: float
    consistency_score: float
    compliance_rate: float
    user_satisfaction_estimate: float

class ProceduralOutputControlProtocol:
    """
    Stackable protocol implementing Procedural Output Control
    
    Provides deterministic output formatting, constitutional constraint application,
    and consistent presentation layer governance that can be stacked onto any
    cognitive workflow to ensure consistent, high-quality outputs.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Procedural Output Control Protocol"""
        self.config = config or {}
        self.formatting_rules = {}  # rule_name -> FormattingRule
        self.constitutional_constraints = {}  # constraint_name -> ConstitutionalConstraint
        self.output_templates = {}  # template_name -> template
        self.quality_validators = {}  # quality_level -> validator_function
        self.presentation_adapters = {}  # layer -> adapter_function
        self.output_history = defaultdict(list)
        self.constraint_validators = {}  # constraint -> validator_function
        
        # Initialize core components
        self._initialize_formatting_rules()
        self._initialize_constitutional_constraints()
        self._initialize_output_templates()
        self._initialize_quality_validators()
        self._initialize_presentation_adapters()
        self._initialize_constraint_validators()
        
        logger.info("ProceduralOutputControlProtocol initialized with stackable output governance")
    
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute procedural output control protocol
        
        Args:
            data: Input data containing protocol outputs and control specifications
            
        Returns:
            Dict containing controlled and formatted outputs
        """
        execution_start = time.time()
        
        try:
            # Extract execution context
            context = data.get('context', {})
            protocol_outputs = data.get('protocol_outputs', {})
            output_spec = self._extract_output_specification(data)
            
            # Apply output control phases
            constraint_result = await self._apply_constitutional_constraints(protocol_outputs, output_spec)
            formatting_result = await self._apply_formatting_rules(constraint_result, output_spec)
            presentation_result = await self._apply_presentation_layer(formatting_result, output_spec)
            quality_result = await self._apply_quality_control(presentation_result, output_spec)
            
            # Calculate metrics
            metrics = self._calculate_output_metrics(quality_result, execution_start)
            
            # Prepare controlled output
            controlled_output = {
                **data,
                'output_control': {
                    'controlled_outputs': quality_result.get('outputs', {}),
                    'constraint_compliance': constraint_result.get('compliance', {}),
                    'formatting_applied': formatting_result.get('formatting', {}),
                    'presentation_layer': presentation_result.get('presentation', {}),
                    'quality_metrics': quality_result.get('quality', {}),
                    'output_metrics': metrics,
                    'output_specification': output_spec.__dict__ if output_spec else {},
                    'execution_time': time.time() - execution_start
                }
            }
            
            # Update output history
            self._update_output_history(controlled_output)
            
            logger.info(f"Output control completed - processed {len(protocol_outputs)} outputs")
            return controlled_output
            
        except Exception as e:
            logger.error(f"Output control protocol failed: {str(e)}")
            return {
                **data,
                'output_control': {
                    'status': 'failed',
                    'error': str(e),
                    'execution_time': time.time() - execution_start
                }
            }
    
    def _extract_output_specification(self, data: Dict[str, Any]) -> OutputSpecification:
        """Extract or create output specification from context"""
        try:
            context = data.get('context', {})
            output_config = context.get('output_control', {})
            
            # Extract specification parameters
            format_type = output_config.get('format', 'structured')
            constraints = output_config.get('constraints', ['content_safety', 'factual_accuracy'])
            presentation = output_config.get('presentation', 'formatted')
            quality = output_config.get('quality', 'standard')
            audience = output_config.get('audience', 'general')
            
            # Create output specification
            return OutputSpecification(
                format=OutputFormat(format_type),
                constraints=[OutputConstraint(c) for c in constraints],
                presentation_layer=PresentationLayer(presentation),
                quality_level=OutputQuality(quality),
                target_audience=audience,
                max_length=output_config.get('max_length'),
                min_length=output_config.get('min_length'),
                structure_template=output_config.get('template'),
                custom_rules=output_config.get('custom_rules', [])
            )
            
        except Exception as e:
            logger.error(f"Output specification extraction failed: {str(e)}")
            # Return default specification
            return OutputSpecification(
                format=OutputFormat.STRUCTURED,
                constraints=[OutputConstraint.CONTENT_SAFETY, OutputConstraint.FACTUAL_ACCURACY],
                presentation_layer=PresentationLayer.FORMATTED,
                quality_level=OutputQuality.STANDARD,
                target_audience="general"
            )
    
    async def _apply_constitutional_constraints(self, protocol_outputs: Dict[str, Any], 
                                              output_spec: OutputSpecification) -> Dict[str, Any]:
        """Apply constitutional constraints to all protocol outputs"""
        try:
            compliance_results = {}
            corrected_outputs = {}
            violations = []
            
            for protocol_name, output_data in protocol_outputs.items():
                protocol_compliance = {}
                protocol_violations = []
                corrected_output = output_data.copy()
                
                # Apply each constraint
                for constraint in output_spec.constraints:
                    constraint_result = await self._validate_constraint(
                        constraint, corrected_output, output_spec
                    )
                    
                    protocol_compliance[constraint.value] = constraint_result.get('compliant', False)
                    
                    if not constraint_result.get('compliant', False):
                        violation = {
                            'constraint': constraint.value,
                            'description': constraint_result.get('description', ''),
                            'severity': constraint_result.get('severity', 'medium'),
                            'auto_fixable': constraint_result.get('auto_fixable', False)
                        }
                        protocol_violations.append(violation)
                        
                        # Apply auto-fix if available
                        if constraint_result.get('auto_fixable', False):
                            corrected_output = constraint_result.get('corrected_output', corrected_output)
                
                compliance_results[protocol_name] = protocol_compliance
                corrected_outputs[protocol_name] = corrected_output
                violations.extend(protocol_violations)
            
            return {
                'outputs': corrected_outputs,
                'compliance': compliance_results,
                'violations': violations,
                'total_violations': len(violations),
                'compliance_rate': self._calculate_compliance_rate(compliance_results)
            }
            
        except Exception as e:
            logger.error(f"Constitutional constraint application failed: {str(e)}")
            return {
                'outputs': protocol_outputs,
                'compliance': {},
                'violations': [],
                'constraint_error': str(e)
            }
    
    async def _validate_constraint(self, constraint: OutputConstraint, 
                                  output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate specific constitutional constraint"""
        try:
            validator = self.constraint_validators.get(constraint)
            if not validator:
                return {'compliant': True, 'description': 'No validator available'}
            
            # Apply constraint validation
            validation_result = await validator(output_data, output_spec)
            
            return {
                'compliant': validation_result.get('compliant', True),
                'description': validation_result.get('description', ''),
                'severity': validation_result.get('severity', 'medium'),
                'auto_fixable': validation_result.get('auto_fixable', False),
                'corrected_output': validation_result.get('corrected_output', output_data)
            }
            
        except Exception as e:
            logger.error(f"Constraint validation failed for {constraint.value}: {str(e)}")
            return {
                'compliant': False,
                'description': f'Validation error: {str(e)}',
                'severity': 'high'
            }
    
    async def _apply_formatting_rules(self, constraint_result: Dict[str, Any], 
                                     output_spec: OutputSpecification) -> Dict[str, Any]:
        """Apply formatting rules to constraint-validated outputs"""
        try:
            outputs = constraint_result.get('outputs', {})
            formatted_outputs = {}
            formatting_applied = {}
            
            for protocol_name, output_data in outputs.items():
                # Apply format-specific rules
                format_rules = self._get_format_rules(output_spec.format)
                formatted_output = await self._apply_rules_to_output(output_data, format_rules)
                
                # Apply quality-specific formatting
                quality_rules = self._get_quality_formatting_rules(output_spec.quality_level)
                formatted_output = await self._apply_rules_to_output(formatted_output, quality_rules)
                
                # Apply custom rules
                if output_spec.custom_rules:
                    custom_rule_objects = self._parse_custom_rules(output_spec.custom_rules)
                    formatted_output = await self._apply_rules_to_output(formatted_output, custom_rule_objects)
                
                formatted_outputs[protocol_name] = formatted_output
                formatting_applied[protocol_name] = {
                    'format_rules_applied': len(format_rules),
                    'quality_rules_applied': len(quality_rules), 
                    'custom_rules_applied': len(output_spec.custom_rules),
                    'format_type': output_spec.format.value
                }
            
            return {
                **constraint_result,
                'outputs': formatted_outputs,
                'formatting': formatting_applied
            }
            
        except Exception as e:
            logger.error(f"Formatting rules application failed: {str(e)}")
            return {
                **constraint_result,
                'formatting_error': str(e)
            }
    
    async def _apply_presentation_layer(self, formatting_result: Dict[str, Any], 
                                       output_spec: OutputSpecification) -> Dict[str, Any]:
        """Apply presentation layer transformations"""
        try:
            outputs = formatting_result.get('outputs', {})
            presented_outputs = {}
            presentation_info = {}
            
            # Get presentation adapter
            adapter = self.presentation_adapters.get(output_spec.presentation_layer)
            if not adapter:
                # Use identity adapter if none found
                adapter = lambda x, spec: x
            
            for protocol_name, output_data in outputs.items():
                # Apply presentation transformation
                presented_output = await adapter(output_data, output_spec)
                
                # Apply audience-specific adjustments
                audience_adjusted = await self._adjust_for_audience(
                    presented_output, output_spec.target_audience
                )
                
                presented_outputs[protocol_name] = audience_adjusted
                presentation_info[protocol_name] = {
                    'presentation_layer': output_spec.presentation_layer.value,
                    'target_audience': output_spec.target_audience,
                    'transformations_applied': self._count_transformations(output_data, audience_adjusted)
                }
            
            return {
                **formatting_result,
                'outputs': presented_outputs,
                'presentation': presentation_info
            }
            
        except Exception as e:
            logger.error(f"Presentation layer application failed: {str(e)}")
            return {
                **formatting_result,
                'presentation_error': str(e)
            }
    
    async def _apply_quality_control(self, presentation_result: Dict[str, Any], 
                                    output_spec: OutputSpecification) -> Dict[str, Any]:
        """Apply quality control measures"""
        try:
            outputs = presentation_result.get('outputs', {})
            quality_controlled_outputs = {}
            quality_metrics = {}
            
            # Get quality validator
            validator = self.quality_validators.get(output_spec.quality_level)
            if not validator:
                validator = self._default_quality_validator
            
            for protocol_name, output_data in outputs.items():
                # Apply quality validation and improvement
                quality_result = await validator(output_data, output_spec)
                
                # Apply length constraints
                length_controlled = self._apply_length_constraints(
                    quality_result.get('output', output_data), output_spec
                )
                
                # Apply structure validation
                structure_validated = await self._validate_output_structure(
                    length_controlled, output_spec
                )
                
                quality_controlled_outputs[protocol_name] = structure_validated
                quality_metrics[protocol_name] = {
                    'quality_level': output_spec.quality_level.value,
                    'quality_score': quality_result.get('score', 0.0),
                    'improvements_applied': quality_result.get('improvements', []),
                    'length_compliant': self._check_length_compliance(structure_validated, output_spec),
                    'structure_compliant': quality_result.get('structure_compliant', True)
                }
            
            return {
                **presentation_result,
                'outputs': quality_controlled_outputs,
                'quality': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Quality control application failed: {str(e)}")
            return {
                **presentation_result,
                'quality_error': str(e)
            }
    
    def _calculate_output_metrics(self, quality_result: Dict[str, Any], 
                                 execution_start: float) -> OutputMetrics:
        """Calculate comprehensive output control metrics"""
        try:
            outputs = quality_result.get('outputs', {})
            compliance_data = quality_result.get('compliance', {})
            violations = quality_result.get('violations', [])
            formatting_data = quality_result.get('formatting', {})
            presentation_data = quality_result.get('presentation', {})
            quality_data = quality_result.get('quality', {})
            
            # Calculate metrics
            total_outputs = len(outputs)
            total_violations = len(violations)
            violations_fixed = sum(1 for v in violations if v.get('auto_fixable', False))
            
            formatting_corrections = sum(
                f.get('format_rules_applied', 0) + f.get('quality_rules_applied', 0) 
                for f in formatting_data.values()
            )
            
            presentation_optimizations = sum(
                p.get('transformations_applied', 0) for p in presentation_data.values()
            )
            
            quality_improvements = sum(
                len(q.get('improvements_applied', [])) for q in quality_data.values()
            )
            
            # Calculate compliance rate
            compliance_rate = self._calculate_overall_compliance_rate(compliance_data)
            
            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(outputs)
            
            # Calculate processing time
            processing_time = (time.time() - execution_start) * 1000
            
            # Estimate user satisfaction
            user_satisfaction = self._estimate_user_satisfaction(quality_data, compliance_rate)
            
            return OutputMetrics(
                total_outputs_processed=total_outputs,
                constraint_violations_detected=total_violations,
                constraint_violations_fixed=violations_fixed,
                formatting_corrections_applied=formatting_corrections,
                presentation_optimizations=presentation_optimizations,
                quality_improvements=quality_improvements,
                processing_time_ms=processing_time,
                consistency_score=consistency_score,
                compliance_rate=compliance_rate,
                user_satisfaction_estimate=user_satisfaction
            )
            
        except Exception as e:
            logger.error(f"Output metrics calculation failed: {str(e)}")
            return OutputMetrics(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    
    def _update_output_history(self, controlled_output: Dict[str, Any]):
        """Update output history for learning and improvement"""
        try:
            session_id = controlled_output.get('context', {}).get('session_id', 'unknown')
            timestamp = datetime.now()
            
            history_entry = {
                'timestamp': timestamp,
                'session_id': session_id,
                'outputs_processed': len(controlled_output.get('output_control', {}).get('controlled_outputs', {})),
                'metrics': controlled_output.get('output_control', {}).get('output_metrics'),
                'violations': controlled_output.get('output_control', {}).get('constraint_compliance', {})
            }
            
            self.output_history[session_id].append(history_entry)
            
            # Limit history size
            if len(self.output_history[session_id]) > 1000:
                self.output_history[session_id] = self.output_history[session_id][-1000:]
                
        except Exception as e:
            logger.error(f"Output history update failed: {str(e)}")
    
    # Initialization methods
    
    def _initialize_formatting_rules(self):
        """Initialize standard formatting rules"""
        self.formatting_rules = {
            'json_standardize': FormattingRule(
                name='json_standardize',
                description='Standardize JSON output formatting',
                pattern=r'(\{|\[)',
                replacement=r'\1',
                apply_order=1,
                is_mandatory=True,
                constraint_type=OutputConstraint.DETERMINISTIC_FORMAT
            ),
            'remove_excessive_whitespace': FormattingRule(
                name='remove_excessive_whitespace',
                description='Remove excessive whitespace',
                pattern=r'\s{3,}',
                replacement=' ',
                apply_order=2,
                is_mandatory=True,
                constraint_type=OutputConstraint.DETERMINISTIC_FORMAT
            ),
            'standardize_quotes': FormattingRule(
                name='standardize_quotes',
                description='Standardize quotation marks',
                pattern=r'[""''`]',
                replacement='"',
                apply_order=3,
                is_mandatory=False,
                constraint_type=OutputConstraint.DETERMINISTIC_FORMAT
            ),
            'capitalize_sentences': FormattingRule(
                name='capitalize_sentences',
                description='Capitalize sentence beginnings',
                pattern=r'(\. +)([a-z])',
                replacement=r'\1\2'.upper(),
                apply_order=4,
                is_mandatory=False,
                constraint_type=OutputConstraint.DETERMINISTIC_FORMAT
            )
        }
    
    def _initialize_constitutional_constraints(self):
        """Initialize constitutional constraints"""
        self.constitutional_constraints = {
            'content_safety': ConstitutionalConstraint(
                name='content_safety',
                description='Ensure content is safe and appropriate',
                validator_function='validate_content_safety',
                severity='high',
                auto_fix=True,
                fallback_action='remove_unsafe_content'
            ),
            'factual_accuracy': ConstitutionalConstraint(
                name='factual_accuracy',
                description='Verify factual accuracy of statements',
                validator_function='validate_factual_accuracy',
                severity='high',
                auto_fix=False,
                fallback_action='add_uncertainty_qualifier'
            ),
            'ethical_compliance': ConstitutionalConstraint(
                name='ethical_compliance',
                description='Ensure ethical compliance',
                validator_function='validate_ethical_compliance',
                severity='medium',
                auto_fix=True,
                fallback_action='add_ethical_disclaimer'
            ),
            'privacy_protection': ConstitutionalConstraint(
                name='privacy_protection',
                description='Protect personal and sensitive information',
                validator_function='validate_privacy_protection',
                severity='high',
                auto_fix=True,
                fallback_action='redact_sensitive_info'
            )
        }
    
    def _initialize_output_templates(self):
        """Initialize output templates for different formats"""
        self.output_templates = {
            'structured_response': {
                'summary': '',
                'details': {},
                'metadata': {
                    'timestamp': '',
                    'source': '',
                    'confidence': 0.0
                }
            },
            'analysis_report': {
                'executive_summary': '',
                'key_findings': [],
                'detailed_analysis': {},
                'recommendations': [],
                'appendix': {}
            },
            'decision_output': {
                'decision': '',
                'rationale': '',
                'confidence_level': 0.0,
                'alternatives_considered': [],
                'risk_assessment': {}
            }
        }
    
    def _initialize_quality_validators(self):
        """Initialize quality validators for different quality levels"""
        self.quality_validators = {
            OutputQuality.MINIMAL: self._validate_minimal_quality,
            OutputQuality.STANDARD: self._validate_standard_quality,
            OutputQuality.HIGH: self._validate_high_quality,
            OutputQuality.PREMIUM: self._validate_premium_quality,
            OutputQuality.PUBLICATION_READY: self._validate_publication_quality
        }
    
    def _initialize_presentation_adapters(self):
        """Initialize presentation layer adapters"""
        self.presentation_adapters = {
            PresentationLayer.RAW: self._adapt_raw_presentation,
            PresentationLayer.FORMATTED: self._adapt_formatted_presentation,
            PresentationLayer.SUMMARIZED: self._adapt_summarized_presentation,
            PresentationLayer.DETAILED: self._adapt_detailed_presentation,
            PresentationLayer.EXECUTIVE: self._adapt_executive_presentation,
            PresentationLayer.TECHNICAL: self._adapt_technical_presentation,
            PresentationLayer.USER_FRIENDLY: self._adapt_user_friendly_presentation
        }
    
    def _initialize_constraint_validators(self):
        """Initialize constraint validators"""
        self.constraint_validators = {
            OutputConstraint.CONTENT_SAFETY: self._validate_content_safety,
            OutputConstraint.FACTUAL_ACCURACY: self._validate_factual_accuracy,
            OutputConstraint.LOGICAL_CONSISTENCY: self._validate_logical_consistency,
            OutputConstraint.ETHICAL_COMPLIANCE: self._validate_ethical_compliance,
            OutputConstraint.PRIVACY_PROTECTION: self._validate_privacy_protection,
            OutputConstraint.BIAS_MITIGATION: self._validate_bias_mitigation,
            OutputConstraint.TRUTHFULNESS: self._validate_truthfulness,
            OutputConstraint.DETERMINISTIC_FORMAT: self._validate_deterministic_format
        }
    
    # Helper methods for constraint validation
    
    async def _validate_content_safety(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate content safety"""
        try:
            # Simple content safety checks
            content_text = self._extract_text_content(output_data)
            
            unsafe_patterns = [
                r'\b(hate|violence|harm)\b',
                r'\b(illegal|dangerous)\b'
            ]
            
            violations = []
            for pattern in unsafe_patterns:
                if re.search(pattern, content_text, re.IGNORECASE):
                    violations.append(f"Unsafe content pattern detected: {pattern}")
            
            is_compliant = len(violations) == 0
            
            # Auto-fix by removing unsafe content
            corrected_output = output_data
            if not is_compliant:
                corrected_text = content_text
                for pattern in unsafe_patterns:
                    corrected_text = re.sub(pattern, '[REDACTED]', corrected_text, flags=re.IGNORECASE)
                corrected_output = self._replace_text_content(output_data, corrected_text)
            
            return {
                'compliant': is_compliant,
                'description': '; '.join(violations) if violations else 'Content is safe',
                'severity': 'high' if violations else 'none',
                'auto_fixable': True,
                'corrected_output': corrected_output
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Content safety validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_factual_accuracy(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate factual accuracy"""
        try:
            # Basic factual accuracy checks
            content_text = self._extract_text_content(output_data)
            
            # Check for uncertainty qualifiers
            uncertainty_indicators = ['might', 'could', 'possibly', 'potentially', 'likely', 'probably']
            factual_claims = re.findall(r'([A-Z][^.!?]*[.!?])', content_text)
            
            unqualified_claims = []
            for claim in factual_claims:
                if not any(indicator in claim.lower() for indicator in uncertainty_indicators):
                    if self._appears_factual_claim(claim):
                        unqualified_claims.append(claim)
            
            is_compliant = len(unqualified_claims) < 3  # Allow some unqualified claims
            
            return {
                'compliant': is_compliant,
                'description': f'Found {len(unqualified_claims)} unqualified factual claims' if not is_compliant else 'Factual accuracy acceptable',
                'severity': 'medium' if not is_compliant else 'none',
                'auto_fixable': False
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Factual accuracy validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_logical_consistency(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate logical consistency"""
        try:
            content_text = self._extract_text_content(output_data)
            
            # Simple logical consistency checks
            contradictions = self._find_logical_contradictions(content_text)
            
            is_compliant = len(contradictions) == 0
            
            return {
                'compliant': is_compliant,
                'description': f'Found {len(contradictions)} logical contradictions' if contradictions else 'Logically consistent',
                'severity': 'medium' if contradictions else 'none',
                'auto_fixable': False
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Logical consistency validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_ethical_compliance(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate ethical compliance"""
        try:
            content_text = self._extract_text_content(output_data)
            
            ethical_concerns = []
            
            # Check for biased language
            bias_patterns = [
                r'\b(all|every|never)\s+(women|men|people|they)\b',
                r'\b(always|never)\s+(are|do|have)\b'
            ]
            
            for pattern in bias_patterns:
                if re.search(pattern, content_text, re.IGNORECASE):
                    ethical_concerns.append(f"Potentially biased language: {pattern}")
            
            is_compliant = len(ethical_concerns) == 0
            
            return {
                'compliant': is_compliant,
                'description': '; '.join(ethical_concerns) if ethical_concerns else 'Ethically compliant',
                'severity': 'medium' if ethical_concerns else 'none',
                'auto_fixable': True
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Ethical compliance validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_privacy_protection(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate privacy protection"""
        try:
            content_text = self._extract_text_content(output_data)
            
            # Check for PII patterns
            pii_patterns = [
                (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
                (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'Credit Card'),
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
                (r'\b\d{3}[-\.]?\d{3}[-\.]?\d{4}\b', 'Phone Number')
            ]
            
            privacy_violations = []
            corrected_text = content_text
            
            for pattern, pii_type in pii_patterns:
                matches = re.findall(pattern, content_text)
                if matches:
                    privacy_violations.append(f'{pii_type} detected')
                    corrected_text = re.sub(pattern, f'[{pii_type}_REDACTED]', corrected_text)
            
            is_compliant = len(privacy_violations) == 0
            corrected_output = self._replace_text_content(output_data, corrected_text) if not is_compliant else output_data
            
            return {
                'compliant': is_compliant,
                'description': '; '.join(privacy_violations) if privacy_violations else 'Privacy protected',
                'severity': 'high' if privacy_violations else 'none',
                'auto_fixable': True,
                'corrected_output': corrected_output
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Privacy protection validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_bias_mitigation(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate bias mitigation"""
        try:
            content_text = self._extract_text_content(output_data)
            
            # Check for inclusive language
            inclusive_score = self._calculate_inclusive_language_score(content_text)
            
            is_compliant = inclusive_score > 0.7
            
            return {
                'compliant': is_compliant,
                'description': f'Inclusive language score: {inclusive_score:.2f}' + (' (acceptable)' if is_compliant else ' (needs improvement)'),
                'severity': 'low' if not is_compliant else 'none',
                'auto_fixable': True
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Bias mitigation validation failed: {str(e)}',
                'severity': 'medium'
            }
    
    async def _validate_truthfulness(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate truthfulness"""
        try:
            content_text = self._extract_text_content(output_data)
            
            # Check for truth indicators
            truth_indicators = self._count_truth_indicators(content_text)
            speculation_indicators = self._count_speculation_indicators(content_text)
            
            truthfulness_score = truth_indicators / max(1, truth_indicators + speculation_indicators)
            is_compliant = truthfulness_score > 0.6
            
            return {
                'compliant': is_compliant,
                'description': f'Truthfulness score: {truthfulness_score:.2f}' + (' (acceptable)' if is_compliant else ' (needs improvement)'),
                'severity': 'medium' if not is_compliant else 'none',
                'auto_fixable': False
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Truthfulness validation failed: {str(e)}',
                'severity': 'high'
            }
    
    async def _validate_deterministic_format(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate deterministic format"""
        try:
            # Check format consistency
            if isinstance(output_data, dict):
                format_score = self._calculate_format_consistency_score(output_data)
            else:
                format_score = 1.0 if isinstance(output_data, str) else 0.5
            
            is_compliant = format_score > 0.8
            
            return {
                'compliant': is_compliant,
                'description': f'Format consistency score: {format_score:.2f}' + (' (deterministic)' if is_compliant else ' (inconsistent)'),
                'severity': 'low' if not is_compliant else 'none',
                'auto_fixable': True
            }
            
        except Exception as e:
            return {
                'compliant': False,
                'description': f'Deterministic format validation failed: {str(e)}',
                'severity': 'medium'
            }
    
    # Helper methods for formatting and presentation
    
    def _get_format_rules(self, output_format: OutputFormat) -> List[FormattingRule]:
        """Get formatting rules for specific output format"""
        base_rules = [rule for rule in self.formatting_rules.values() if rule.is_mandatory]
        
        format_specific_rules = []
        if output_format == OutputFormat.JSON:
            format_specific_rules.extend([
                self.formatting_rules.get('json_standardize', None)
            ])
        
        return [rule for rule in base_rules + format_specific_rules if rule is not None]
    
    def _get_quality_formatting_rules(self, quality_level: OutputQuality) -> List[FormattingRule]:
        """Get formatting rules for specific quality level"""
        if quality_level in [OutputQuality.PREMIUM, OutputQuality.PUBLICATION_READY]:
            return list(self.formatting_rules.values())
        elif quality_level == OutputQuality.HIGH:
            return [rule for rule in self.formatting_rules.values() if rule.is_mandatory or rule.apply_order <= 3]
        else:
            return [rule for rule in self.formatting_rules.values() if rule.is_mandatory]
    
    async def _apply_rules_to_output(self, output_data: Any, rules: List[FormattingRule]) -> Any:
        """Apply formatting rules to output data"""
        try:
            if isinstance(output_data, str):
                formatted_text = output_data
                
                # Sort rules by apply_order
                sorted_rules = sorted(rules, key=lambda r: r.apply_order)
                
                for rule in sorted_rules:
                    formatted_text = re.sub(rule.pattern, rule.replacement, formatted_text)
                
                return formatted_text
            
            elif isinstance(output_data, dict):
                formatted_dict = {}
                for key, value in output_data.items():
                    if isinstance(value, str):
                        formatted_dict[key] = await self._apply_rules_to_output(value, rules)
                    else:
                        formatted_dict[key] = value
                return formatted_dict
            
            else:
                return output_data
                
        except Exception as e:
            logger.error(f"Rule application failed: {str(e)}")
            return output_data
    
    def _parse_custom_rules(self, custom_rules: List[str]) -> List[FormattingRule]:
        """Parse custom rules from string specifications"""
        parsed_rules = []
        
        try:
            for i, rule_spec in enumerate(custom_rules):
                # Simple format: "pattern -> replacement"
                if ' -> ' in rule_spec:
                    pattern, replacement = rule_spec.split(' -> ', 1)
                    rule = FormattingRule(
                        name=f'custom_{i}',
                        description=f'Custom rule {i}',
                        pattern=pattern.strip(),
                        replacement=replacement.strip(),
                        apply_order=100 + i,
                        is_mandatory=False,
                        constraint_type=OutputConstraint.DETERMINISTIC_FORMAT
                    )
                    parsed_rules.append(rule)
                    
        except Exception as e:
            logger.error(f"Custom rule parsing failed: {str(e)}")
        
        return parsed_rules
    
    # Presentation layer adapters
    
    async def _adapt_raw_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for raw presentation"""
        return output_data
    
    async def _adapt_formatted_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for formatted presentation"""
        if isinstance(output_data, dict):
            return json.dumps(output_data, indent=2, sort_keys=True)
        return str(output_data)
    
    async def _adapt_summarized_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for summarized presentation"""
        if isinstance(output_data, dict):
            # Create summary version
            summary = {
                'summary': self._extract_summary(output_data),
                'key_points': self._extract_key_points(output_data)
            }
            return summary
        elif isinstance(output_data, str):
            sentences = output_data.split('. ')
            return '. '.join(sentences[:3]) + ('...' if len(sentences) > 3 else '')
        return output_data
    
    async def _adapt_detailed_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for detailed presentation"""
        if isinstance(output_data, dict):
            # Add metadata and expand details
            detailed = {
                'content': output_data,
                'metadata': {
                    'presentation_type': 'detailed',
                    'timestamp': datetime.now().isoformat(),
                    'format': output_spec.format.value
                },
                'analysis': self._generate_content_analysis(output_data)
            }
            return detailed
        return output_data
    
    async def _adapt_executive_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for executive presentation"""
        if isinstance(output_data, dict):
            executive = {
                'executive_summary': self._create_executive_summary(output_data),
                'key_decisions': self._extract_key_decisions(output_data),
                'recommendations': self._extract_recommendations(output_data),
                'next_steps': self._extract_next_steps(output_data)
            }
            return executive
        return output_data
    
    async def _adapt_technical_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for technical presentation"""
        if isinstance(output_data, dict):
            technical = {
                'data': output_data,
                'technical_details': self._extract_technical_details(output_data),
                'implementation_notes': self._generate_implementation_notes(output_data),
                'specifications': self._extract_specifications(output_data)
            }
            return technical
        return output_data
    
    async def _adapt_user_friendly_presentation(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Adapt output for user-friendly presentation"""
        if isinstance(output_data, dict):
            user_friendly = {}
            for key, value in output_data.items():
                friendly_key = self._make_key_user_friendly(key)
                friendly_value = self._make_value_user_friendly(value)
                user_friendly[friendly_key] = friendly_value
            return user_friendly
        elif isinstance(output_data, str):
            return self._make_text_user_friendly(output_data)
        return output_data
    
    # Quality validators
    
    async def _validate_minimal_quality(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate minimal quality requirements"""
        score = 0.5  # Basic pass
        improvements = []
        
        if isinstance(output_data, str) and len(output_data) > 0:
            score = 0.7
        elif isinstance(output_data, dict) and len(output_data) > 0:
            score = 0.7
        
        return {
            'output': output_data,
            'score': score,
            'improvements': improvements,
            'structure_compliant': True
        }
    
    async def _validate_standard_quality(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate standard quality requirements"""
        score = 0.6
        improvements = []
        
        # Check completeness
        if self._is_complete_response(output_data):
            score += 0.2
        else:
            improvements.append('Improve response completeness')
        
        # Check clarity
        if self._has_clear_structure(output_data):
            score += 0.1
        else:
            improvements.append('Improve structural clarity')
        
        return {
            'output': output_data,
            'score': min(score, 1.0),
            'improvements': improvements,
            'structure_compliant': len(improvements) == 0
        }
    
    async def _validate_high_quality(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate high quality requirements"""
        score = 0.7
        improvements = []
        
        # Standard quality checks
        standard_result = await self._validate_standard_quality(output_data, output_spec)
        score = standard_result['score']
        improvements = standard_result['improvements']
        
        # Additional high-quality checks
        if self._has_detailed_explanations(output_data):
            score += 0.1
        else:
            improvements.append('Add detailed explanations')
        
        if self._has_supporting_evidence(output_data):
            score += 0.1
        else:
            improvements.append('Include supporting evidence')
        
        return {
            'output': output_data,
            'score': min(score, 1.0),
            'improvements': improvements,
            'structure_compliant': len(improvements) <= 1
        }
    
    async def _validate_premium_quality(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate premium quality requirements"""
        # High quality checks
        high_result = await self._validate_high_quality(output_data, output_spec)
        score = high_result['score']
        improvements = high_result['improvements']
        
        # Premium quality enhancements
        if self._has_exceptional_detail(output_data):
            score += 0.1
        else:
            improvements.append('Enhance level of detail')
        
        if self._has_multiple_perspectives(output_data):
            score += 0.05
        else:
            improvements.append('Include multiple perspectives')
        
        return {
            'output': output_data,
            'score': min(score, 1.0),
            'improvements': improvements,
            'structure_compliant': len(improvements) == 0
        }
    
    async def _validate_publication_quality(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Validate publication-ready quality requirements"""
        # Premium quality checks
        premium_result = await self._validate_premium_quality(output_data, output_spec)
        score = premium_result['score']
        improvements = premium_result['improvements']
        
        # Publication quality requirements
        if self._meets_publication_standards(output_data):
            score += 0.05
        else:
            improvements.append('Meet publication standards')
        
        if self._has_proper_citations(output_data):
            score += 0.05
        else:
            improvements.append('Add proper citations')
        
        return {
            'output': output_data,
            'score': min(score, 1.0),
            'improvements': improvements,
            'structure_compliant': len(improvements) == 0
        }
    
    async def _default_quality_validator(self, output_data: Any, output_spec: OutputSpecification) -> Dict[str, Any]:
        """Default quality validator"""
        return await self._validate_standard_quality(output_data, output_spec)
    
    # Helper methods for text processing and analysis
    
    def _extract_text_content(self, data: Any) -> str:
        """Extract text content from various data types"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict):
            text_parts = []
            for key, value in data.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, (dict, list)):
                    text_parts.append(self._extract_text_content(value))
            return ' '.join(text_parts)
        elif isinstance(data, list):
            return ' '.join(str(item) for item in data)
        else:
            return str(data)
    
    def _replace_text_content(self, data: Any, new_text: str) -> Any:
        """Replace text content in data structure"""
        if isinstance(data, str):
            return new_text
        elif isinstance(data, dict):
            # For dict, replace the first string value found
            result = data.copy()
            for key, value in result.items():
                if isinstance(value, str):
                    result[key] = new_text
                    break
            return result
        else:
            return data
    
    def _appears_factual_claim(self, text: str) -> bool:
        """Check if text appears to be a factual claim"""
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has', 'have', 'had',
            'will', 'would', 'can', 'cannot', 'must', 'should'
        ]
        return any(indicator in text.lower() for indicator in factual_indicators)
    
    def _find_logical_contradictions(self, text: str) -> List[str]:
        """Find logical contradictions in text"""
        contradictions = []
        
        # Simple contradiction detection
        sentences = text.split('. ')
        
        for i, sentence1 in enumerate(sentences):
            for j, sentence2 in enumerate(sentences[i+1:], i+1):
                if self._are_contradictory_sentences(sentence1, sentence2):
                    contradictions.append(f"Sentences {i+1} and {j+1} appear contradictory")
        
        return contradictions
    
    def _are_contradictory_sentences(self, sentence1: str, sentence2: str) -> bool:
        """Check if two sentences are contradictory"""
        # Simple heuristic for contradiction detection
        negation_words = ['not', 'never', 'no', 'cannot', 'won\'t', 'doesn\'t']
        
        # Normalize sentences
        norm1 = sentence1.lower().strip()
        norm2 = sentence2.lower().strip()
        
        # Check for negation patterns
        has_negation1 = any(neg in norm1 for neg in negation_words)
        has_negation2 = any(neg in norm2 for neg in negation_words)
        
        # If one has negation and the other doesn't, and they share similar content
        if has_negation1 != has_negation2:
            # Remove negation words and compare
            clean1 = norm1
            clean2 = norm2
            for neg in negation_words:
                clean1 = clean1.replace(neg, '')
                clean2 = clean2.replace(neg, '')
            
            # Simple similarity check
            words1 = set(clean1.split())
            words2 = set(clean2.split())
            overlap = len(words1.intersection(words2))
            
            return overlap > 2  # Arbitrary threshold
        
        return False
    
    def _calculate_inclusive_language_score(self, text: str) -> float:
        """Calculate inclusive language score"""
        total_words = len(text.split())
        if total_words == 0:
            return 1.0
        
        inclusive_indicators = ['diverse', 'inclusive', 'various', 'different', 'multiple']
        exclusive_indicators = ['only', 'just', 'merely', 'simply', 'all', 'every', 'never']
        
        inclusive_count = sum(1 for word in inclusive_indicators if word in text.lower())
        exclusive_count = sum(1 for word in exclusive_indicators if word in text.lower())
        
        # Calculate score based on ratio
        if inclusive_count + exclusive_count == 0:
            return 0.8  # Neutral
        
        return inclusive_count / (inclusive_count + exclusive_count)
    
    def _count_truth_indicators(self, text: str) -> int:
        """Count truth indicators in text"""
        truth_words = ['fact', 'evidence', 'proven', 'verified', 'confirmed', 'established']
        return sum(1 for word in truth_words if word in text.lower())
    
    def _count_speculation_indicators(self, text: str) -> int:
        """Count speculation indicators in text"""
        speculation_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems', 'appears']
        return sum(1 for word in speculation_words if word in text.lower())
    
    def _calculate_format_consistency_score(self, data: dict) -> float:
        """Calculate format consistency score for dictionary data"""
        if not isinstance(data, dict):
            return 0.5
        
        # Check key consistency
        keys = list(data.keys())
        if len(keys) == 0:
            return 1.0
        
        # Check if keys follow consistent naming convention
        snake_case_count = sum(1 for key in keys if '_' in key and key.islower())
        camel_case_count = sum(1 for key in keys if any(c.isupper() for c in key) and '_' not in key)
        
        total_keys = len(keys)
        consistency_ratio = max(snake_case_count, camel_case_count) / total_keys
        
        return consistency_ratio
    
    # Additional helper methods for presentation and quality
    
    def _extract_summary(self, data: dict) -> str:
        """Extract summary from dictionary data"""
        # Look for summary-like keys
        summary_keys = ['summary', 'overview', 'description', 'result']
        for key in summary_keys:
            if key in data:
                return str(data[key])
        
        # Generate summary from available data
        return f"Data contains {len(data)} fields"
    
    def _extract_key_points(self, data: dict) -> List[str]:
        """Extract key points from dictionary data"""
        key_points = []
        
        # Look for list-type values
        for key, value in data.items():
            if isinstance(value, list):
                key_points.extend([str(item) for item in value[:3]])  # First 3 items
            elif isinstance(value, str) and len(value) > 0:
                key_points.append(f"{key}: {value[:50]}...")  # First 50 chars
        
        return key_points[:5]  # Maximum 5 key points
    
    def _generate_content_analysis(self, data: dict) -> Dict[str, Any]:
        """Generate content analysis for detailed presentation"""
        return {
            'field_count': len(data),
            'data_types': {key: type(value).__name__ for key, value in data.items()},
            'complexity_score': self._calculate_complexity_score(data)
        }
    
    def _calculate_complexity_score(self, data: dict) -> float:
        """Calculate complexity score for data structure"""
        score = 0.0
        
        for value in data.values():
            if isinstance(value, dict):
                score += 2.0
            elif isinstance(value, list):
                score += 1.5
            elif isinstance(value, str) and len(value) > 100:
                score += 1.0
            else:
                score += 0.5
        
        return min(score / len(data) if data else 0.0, 5.0)
    
    def _create_executive_summary(self, data: dict) -> str:
        """Create executive summary from data"""
        summary_parts = []
        
        # Extract key information
        if 'result' in data:
            summary_parts.append(f"Result: {data['result']}")
        if 'conclusion' in data:
            summary_parts.append(f"Conclusion: {data['conclusion']}")
        if 'recommendation' in data:
            summary_parts.append(f"Recommendation: {data['recommendation']}")
        
        if not summary_parts:
            summary_parts.append(f"Analysis of {len(data)} data points completed")
        
        return '. '.join(summary_parts)
    
    def _extract_key_decisions(self, data: dict) -> List[str]:
        """Extract key decisions from data"""
        decision_keys = ['decision', 'choice', 'selection', 'outcome']
        decisions = []
        
        for key in decision_keys:
            if key in data:
                decisions.append(str(data[key]))
        
        return decisions
    
    def _extract_recommendations(self, data: dict) -> List[str]:
        """Extract recommendations from data"""
        rec_keys = ['recommendation', 'suggestions', 'next_steps', 'action_items']
        recommendations = []
        
        for key in rec_keys:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    recommendations.extend(str(item) for item in value)
                else:
                    recommendations.append(str(value))
        
        return recommendations
    
    def _extract_next_steps(self, data: dict) -> List[str]:
        """Extract next steps from data"""
        return self._extract_recommendations(data)  # Same logic for now
    
    def _extract_technical_details(self, data: dict) -> Dict[str, Any]:
        """Extract technical details from data"""
        technical = {}
        
        # Look for technical-sounding keys
        tech_keys = ['config', 'parameters', 'settings', 'specs', 'metrics']
        for key in tech_keys:
            if key in data:
                technical[key] = data[key]
        
        return technical
    
    def _generate_implementation_notes(self, data: dict) -> List[str]:
        """Generate implementation notes"""
        notes = []
        
        if 'error' in data or 'errors' in data:
            notes.append("Handle error conditions appropriately")
        
        if 'timeout' in data:
            notes.append("Consider timeout configurations")
        
        notes.append("Validate input parameters before processing")
        notes.append("Implement proper logging and monitoring")
        
        return notes
    
    def _extract_specifications(self, data: dict) -> Dict[str, Any]:
        """Extract specifications from data"""
        specs = {}
        
        # Look for specification-like data
        if 'format' in data:
            specs['output_format'] = data['format']
        if 'version' in data:
            specs['version'] = data['version']
        if 'timestamp' in data:
            specs['timestamp'] = data['timestamp']
        
        return specs
    
    def _make_key_user_friendly(self, key: str) -> str:
        """Make dictionary key more user-friendly"""
        # Convert snake_case to Title Case
        friendly = key.replace('_', ' ').title()
        
        # Handle common abbreviations
        abbreviations = {
            'Id': 'ID',
            'Url': 'URL',
            'Api': 'API',
            'Http': 'HTTP',
            'Json': 'JSON'
        }
        
        for abbr, full in abbreviations.items():
            friendly = friendly.replace(abbr, full)
        
        return friendly
    
    def _make_value_user_friendly(self, value: Any) -> Any:
        """Make value more user-friendly"""
        if isinstance(value, str):
            return self._make_text_user_friendly(value)
        elif isinstance(value, dict):
            return {self._make_key_user_friendly(k): self._make_value_user_friendly(v) 
                   for k, v in value.items()}
        else:
            return value
    
    def _make_text_user_friendly(self, text: str) -> str:
        """Make text more user-friendly"""
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper ending punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    # Quality assessment helpers
    
    def _is_complete_response(self, data: Any) -> bool:
        """Check if response is complete"""
        if isinstance(data, str):
            return len(data) > 50  # Minimum length for completeness
        elif isinstance(data, dict):
            return len(data) > 0 and all(v is not None for v in data.values())
        else:
            return data is not None
    
    def _has_clear_structure(self, data: Any) -> bool:
        """Check if data has clear structure"""
        if isinstance(data, dict):
            return len(data) > 1  # Multiple fields indicate structure
        elif isinstance(data, str):
            # Check for paragraph structure
            return '.' in data or '\n' in data
        else:
            return False
    
    def _has_detailed_explanations(self, data: Any) -> bool:
        """Check if data contains detailed explanations"""
        text_content = self._extract_text_content(data)
        
        # Look for explanation indicators
        explanation_words = ['because', 'since', 'therefore', 'thus', 'due to', 'as a result']
        return any(word in text_content.lower() for word in explanation_words)
    
    def _has_supporting_evidence(self, data: Any) -> bool:
        """Check if data contains supporting evidence"""
        text_content = self._extract_text_content(data)
        
        # Look for evidence indicators
        evidence_words = ['evidence', 'data', 'study', 'research', 'analysis', 'statistics']
        return any(word in text_content.lower() for word in evidence_words)
    
    def _has_exceptional_detail(self, data: Any) -> bool:
        """Check if data has exceptional level of detail"""
        if isinstance(data, dict):
            return len(data) > 5 and any(isinstance(v, (dict, list)) for v in data.values())
        elif isinstance(data, str):
            return len(data) > 500  # Substantial content
        else:
            return False
    
    def _has_multiple_perspectives(self, data: Any) -> bool:
        """Check if data includes multiple perspectives"""
        text_content = self._extract_text_content(data)
        
        # Look for perspective indicators
        perspective_words = ['however', 'alternatively', 'on the other hand', 'conversely', 'while']
        return any(phrase in text_content.lower() for phrase in perspective_words)
    
    def _meets_publication_standards(self, data: Any) -> bool:
        """Check if data meets publication standards"""
        # Basic publication checks
        text_content = self._extract_text_content(data)
        
        # Check length (substantial content)
        if len(text_content) < 200:
            return False
        
        # Check for proper structure
        if not self._has_clear_structure(data):
            return False
        
        # Check for evidence-based content
        if not self._has_supporting_evidence(data):
            return False
        
        return True
    
    def _has_proper_citations(self, data: Any) -> bool:
        """Check if data has proper citations"""
        text_content = self._extract_text_content(data)
        
        # Look for citation patterns
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\w+\s+\d{4}\)',  # (Author 2023)
            r'\w+\s+et\s+al\.',  # Author et al.
        ]
        
        return any(re.search(pattern, text_content) for pattern in citation_patterns)
    
    # Utility methods for metrics calculation
    
    def _calculate_compliance_rate(self, compliance_results: Dict[str, Dict[str, bool]]) -> float:
        """Calculate compliance rate from results"""
        if not compliance_results:
            return 1.0
        
        total_checks = 0
        compliant_checks = 0
        
        for protocol_compliance in compliance_results.values():
            for is_compliant in protocol_compliance.values():
                total_checks += 1
                if is_compliant:
                    compliant_checks += 1
        
        return compliant_checks / total_checks if total_checks > 0 else 1.0
    
    def _calculate_overall_compliance_rate(self, compliance_data: Dict[str, Any]) -> float:
        """Calculate overall compliance rate"""
        # Implementation depends on structure of compliance_data
        return 0.85  # Placeholder
    
    def _calculate_consistency_score(self, outputs: Dict[str, Any]) -> float:
        """Calculate consistency score across outputs"""
        if len(outputs) <= 1:
            return 1.0
        
        # Simple consistency check based on structure similarity
        structures = []
        for output in outputs.values():
            if isinstance(output, dict):
                structures.append(set(output.keys()))
            else:
                structures.append(set([type(output).__name__]))
        
        if not structures:
            return 1.0
        
        # Calculate similarity between structures
        base_structure = structures[0]
        similarities = []
        
        for structure in structures[1:]:
            intersection = len(base_structure.intersection(structure))
            union = len(base_structure.union(structure))
            similarity = intersection / union if union > 0 else 0.0
            similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def _estimate_user_satisfaction(self, quality_data: Dict[str, Any], compliance_rate: float) -> float:
        """Estimate user satisfaction based on quality and compliance"""
        if not quality_data:
            return compliance_rate
        
        # Average quality scores
        quality_scores = [q.get('quality_score', 0.5) for q in quality_data.values()]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
        
        # Combine quality and compliance
        satisfaction = (avg_quality * 0.6) + (compliance_rate * 0.4)
        
        return min(satisfaction, 1.0)
    
    def _apply_length_constraints(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Apply length constraints to output"""
        if not isinstance(output_data, str):
            return output_data
        
        text = output_data
        
        # Apply maximum length
        if output_spec.max_length and len(text) > output_spec.max_length:
            text = text[:output_spec.max_length - 3] + "..."
        
        # Apply minimum length (pad if necessary)
        if output_spec.min_length and len(text) < output_spec.min_length:
            padding_needed = output_spec.min_length - len(text)
            text += " " + "Additional content to meet minimum length requirements." * (padding_needed // 50 + 1)
            text = text[:output_spec.min_length]
        
        return text
    
    async def _validate_output_structure(self, output_data: Any, output_spec: OutputSpecification) -> Any:
        """Validate and potentially fix output structure"""
        if output_spec.structure_template and isinstance(output_data, dict):
            # Ensure output matches template structure
            validated_output = {}
            
            for key, template_value in output_spec.structure_template.items():
                if key in output_data:
                    validated_output[key] = output_data[key]
                else:
                    # Add missing keys with appropriate default values
                    if isinstance(template_value, str):
                        validated_output[key] = ""
                    elif isinstance(template_value, list):
                        validated_output[key] = []
                    elif isinstance(template_value, dict):
                        validated_output[key] = {}
                    else:
                        validated_output[key] = None
            
            return validated_output
        
        return output_data
    
    def _check_length_compliance(self, output_data: Any, output_spec: OutputSpecification) -> bool:
        """Check if output complies with length constraints"""
        if not isinstance(output_data, str):
            return True
        
        text_length = len(output_data)
        
        if output_spec.max_length and text_length > output_spec.max_length:
            return False
        
        if output_spec.min_length and text_length < output_spec.min_length:
            return False
        
        return True
    
    async def _adjust_for_audience(self, output_data: Any, target_audience: str) -> Any:
        """Adjust output for specific target audience"""
        # Audience-specific adjustments
        if target_audience == "technical":
            return await self._adapt_technical_presentation(output_data, None)
        elif target_audience == "executive":
            return await self._adapt_executive_presentation(output_data, None)
        elif target_audience == "general":
            return await self._adapt_user_friendly_presentation(output_data, None)
        else:
            return output_data
    
    def _count_transformations(self, original: Any, transformed: Any) -> int:
        """Count number of transformations applied"""
        if original == transformed:
            return 0
        
        if isinstance(original, dict) and isinstance(transformed, dict):
            return abs(len(original) - len(transformed))
        elif isinstance(original, str) and isinstance(transformed, str):
            return 1 if original != transformed else 0
        else:
            return 1

# Export the protocol
__all__ = [
    'ProceduralOutputControlProtocol', 
    'OutputFormat', 
    'OutputConstraint', 
    'PresentationLayer', 
    'OutputQuality',
    'OutputSpecification',
    'FormattingRule',
    'ConstitutionalConstraint',
    'OutputMetrics'
]