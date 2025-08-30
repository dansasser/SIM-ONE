"""
Law 5: Deterministic Reliability Protocol
"Governed systems must produce consistent, predictable outcomes rather than probabilistic variations."

This stackable protocol validates that cognitive systems exhibit deterministic, reliable behavior
with consistent outputs for identical inputs, avoiding probabilistic variations.
"""
import logging
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import statistics

logger = logging.getLogger(__name__)

class ReliabilityMetric(Enum):
    """Types of reliability metrics"""
    OUTPUT_CONSISTENCY = "output_consistency"
    BEHAVIORAL_DETERMINISM = "behavioral_determinism"
    ERROR_PREDICTABILITY = "error_predictability"
    RESPONSE_STABILITY = "response_stability"
    EXECUTION_REPEATABILITY = "execution_repeatability"

class ConsistencyLevel(Enum):
    """Levels of consistency"""
    PERFECT = "perfect"
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    UNRELIABLE = "unreliable"

@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability and determinism metrics"""
    output_consistency_score: float
    behavioral_determinism_score: float
    execution_repeatability_score: float
    response_stability_score: float
    error_predictability_score: float
    probabilistic_variation_count: int
    consistency_variance: float
    deterministic_behavior_ratio: float
    reliability_confidence: float
    overall_predictability_score: float

class DeterministicReliabilityProtocol:
    """
    Stackable protocol implementing Law 5: Deterministic Reliability
    
    Ensures cognitive systems produce consistent, predictable outcomes
    rather than probabilistic variations or unreliable behavior.
    """
    
    def __init__(self):
        self.reliability_requirements = {
            "minimum_output_consistency": 0.85,
            "minimum_behavioral_determinism": 0.8,
            "minimum_repeatability": 0.9,
            "maximum_probabilistic_variation": 0.2,
            "minimum_response_stability": 0.8,
            "minimum_error_predictability": 0.7
        }
        
        # Patterns indicating probabilistic behavior (violations)
        self.probabilistic_indicators = [
            r"\b(?:random|randomly|chance|probability|maybe|perhaps)\b",
            r"\b(?:might|could|may|possibly|potentially)\b",
            r"\b(?:sometimes|occasionally|often|usually|typically)\b",
            r"\b(?:varies|variable|inconsistent|unpredictable)\b",
            r"\b(?:approximately|roughly|around|about|estimate)\b"
        ]
        
        # Patterns indicating deterministic behavior (positive indicators)
        self.deterministic_indicators = [
            r"\b(?:always|never|consistently|invariably|definitely)\b",
            r"\b(?:exactly|precisely|specifically|determined|fixed)\b",
            r"\b(?:guaranteed|certain|sure|definite|absolute)\b",
            r"\b(?:predictable|reliable|stable|constant|repeatable)\b",
            r"\b(?:systematic|methodical|structured|algorithmic)\b"
        ]
        
        # Error patterns that should be predictable
        self.error_patterns = [
            "validation_error", "consistency_error", "logic_error",
            "input_error", "constraint_violation", "format_error"
        ]
        
    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Law 5 validation: Deterministic Reliability
        
        Args:
            data: Execution context containing outputs, execution history, and consistency data
            
        Returns:
            Validation results for deterministic reliability compliance
        """
        logger.info("Executing Law 5: Deterministic Reliability validation")
        start_time = time.time()
        
        # Extract relevant data
        cognitive_outputs = data.get("cognitive_outputs", {})
        execution_history = data.get("execution_history", [])
        protocol_stack = data.get("protocol_stack", [])
        execution_metrics = data.get("execution_metrics", {})
        previous_results = data.get("previous_results", [])
        input_variations = data.get("input_variations", {})
        
        # Calculate reliability metrics
        reliability_metrics = self._calculate_reliability_metrics(
            cognitive_outputs, execution_history, previous_results, input_variations
        )
        
        # Analyze output consistency
        consistency_analysis = self._analyze_output_consistency(
            cognitive_outputs, previous_results, input_variations
        )
        
        # Assess behavioral determinism
        determinism_assessment = self._assess_behavioral_determinism(
            execution_history, protocol_stack, cognitive_outputs
        )
        
        # Check execution repeatability
        repeatability_check = self._check_execution_repeatability(
            execution_history, previous_results, input_variations
        )
        
        # Evaluate response stability
        stability_evaluation = self._evaluate_response_stability(
            cognitive_outputs, execution_history, execution_metrics
        )
        
        # Detect probabilistic variations
        probabilistic_detection = self._detect_probabilistic_variations(
            cognitive_outputs, execution_history
        )
        
        # Assess error predictability
        error_predictability = self._assess_error_predictability(
            execution_history, execution_metrics
        )
        
        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(
            reliability_metrics, consistency_analysis, determinism_assessment,
            probabilistic_detection
        )
        
        # Identify violations
        violations = self._identify_violations(
            reliability_metrics, probabilistic_detection, consistency_analysis
        )
        
        execution_time = time.time() - start_time
        
        result = {
            "law": "Law5_DeterministicReliability",
            "compliance_score": compliance_score,
            "reliability_metrics": self._metrics_to_dict(reliability_metrics),
            "consistency_analysis": consistency_analysis,
            "determinism_assessment": determinism_assessment,
            "repeatability_check": repeatability_check,
            "stability_evaluation": stability_evaluation,
            "probabilistic_detection": probabilistic_detection,
            "error_predictability": error_predictability,
            "violations": violations,
            "execution_time": execution_time,
            "recommendations": self._generate_recommendations(reliability_metrics, violations),
            "status": "compliant" if compliance_score >= 0.8 else "non_compliant"
        }
        
        logger.info(f"Law 5 validation completed: {result['status']} (score: {compliance_score:.3f})")
        return result
    
    def _calculate_reliability_metrics(self, 
                                     cognitive_outputs: Dict[str, Any],
                                     execution_history: List[Dict[str, Any]],
                                     previous_results: List[Dict[str, Any]],
                                     input_variations: Dict[str, Any]) -> ReliabilityMetrics:
        """Calculate comprehensive reliability and determinism metrics"""
        
        # Output consistency score
        output_consistency = self._calculate_output_consistency(
            cognitive_outputs, previous_results
        )
        
        # Behavioral determinism score
        behavioral_determinism = self._calculate_behavioral_determinism(
            execution_history, cognitive_outputs
        )
        
        # Execution repeatability score
        execution_repeatability = self._calculate_execution_repeatability(
            execution_history, input_variations
        )
        
        # Response stability score
        response_stability = self._calculate_response_stability(
            cognitive_outputs, execution_history
        )
        
        # Error predictability score
        error_predictability = self._calculate_error_predictability(
            execution_history
        )
        
        # Count probabilistic variations
        probabilistic_count = self._count_probabilistic_variations(
            cognitive_outputs, execution_history
        )
        
        # Calculate consistency variance
        consistency_variance = self._calculate_consistency_variance(
            previous_results, cognitive_outputs
        )
        
        # Calculate deterministic behavior ratio
        deterministic_ratio = self._calculate_deterministic_behavior_ratio(
            cognitive_outputs, execution_history
        )
        
        # Calculate reliability confidence
        reliability_confidence = self._calculate_reliability_confidence(
            output_consistency, behavioral_determinism, execution_repeatability
        )
        
        # Calculate overall predictability score
        predictability_score = self._calculate_overall_predictability(
            output_consistency, behavioral_determinism, response_stability, error_predictability
        )
        
        return ReliabilityMetrics(
            output_consistency_score=output_consistency,
            behavioral_determinism_score=behavioral_determinism,
            execution_repeatability_score=execution_repeatability,
            response_stability_score=response_stability,
            error_predictability_score=error_predictability,
            probabilistic_variation_count=probabilistic_count,
            consistency_variance=consistency_variance,
            deterministic_behavior_ratio=deterministic_ratio,
            reliability_confidence=reliability_confidence,
            overall_predictability_score=predictability_score
        )
    
    def _calculate_output_consistency(self, 
                                    cognitive_outputs: Dict[str, Any],
                                    previous_results: List[Dict[str, Any]]) -> float:
        """Calculate output consistency across multiple executions"""
        
        if not previous_results:
            return 0.8  # Default moderate consistency without comparison data
        
        consistency_scores = []
        
        # Compare current outputs with previous results
        for prev_result in previous_results[-5:]:  # Last 5 results for comparison
            if isinstance(prev_result, dict):
                similarity_score = self._calculate_output_similarity(
                    cognitive_outputs, prev_result
                )
                consistency_scores.append(similarity_score)
        
        if consistency_scores:
            return sum(consistency_scores) / len(consistency_scores)
        else:
            return 0.8
    
    def _calculate_output_similarity(self, 
                                   output1: Dict[str, Any], 
                                   output2: Dict[str, Any]) -> float:
        """Calculate similarity between two output sets"""
        
        if not output1 or not output2:
            return 0.0
        
        similarity_factors = []
        
        # Compare common keys
        common_keys = set(output1.keys()).intersection(set(output2.keys()))
        
        for key in common_keys:
            val1, val2 = output1[key], output2[key]
            
            if isinstance(val1, str) and isinstance(val2, str):
                # Text similarity
                text_similarity = self._calculate_text_similarity(val1, val2)
                similarity_factors.append(text_similarity)
                
            elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                if val1 == val2:
                    similarity_factors.append(1.0)
                else:
                    max_val = max(abs(val1), abs(val2), 1.0)
                    diff_ratio = abs(val1 - val2) / max_val
                    similarity_factors.append(max(0.0, 1.0 - diff_ratio))
                    
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # Nested dictionary similarity
                nested_similarity = self._calculate_output_similarity(val1, val2)
                similarity_factors.append(nested_similarity)
                
            elif type(val1) == type(val2):
                # Same type comparison
                similarity_factors.append(1.0 if val1 == val2 else 0.5)
            else:
                # Different types
                similarity_factors.append(0.0)
        
        # Penalty for missing keys
        all_keys = set(output1.keys()).union(set(output2.keys()))
        key_coverage = len(common_keys) / len(all_keys) if all_keys else 1.0
        
        if similarity_factors:
            avg_similarity = sum(similarity_factors) / len(similarity_factors)
            return avg_similarity * key_coverage
        else:
            return key_coverage
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        
        if text1 == text2:
            return 1.0
        
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1_norm = text1.lower().strip()
        text2_norm = text2.lower().strip()
        
        # Calculate word overlap
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Consider length similarity
        length_ratio = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        return (jaccard_similarity * 0.7) + (length_ratio * 0.3)
    
    def _calculate_behavioral_determinism(self, 
                                        execution_history: List[Dict[str, Any]],
                                        cognitive_outputs: Dict[str, Any]) -> float:
        """Calculate behavioral determinism score"""
        
        determinism_score = 0.7  # Base score
        
        # Analyze execution patterns for deterministic behavior
        if execution_history:
            pattern_consistency = self._analyze_execution_patterns(execution_history)
            determinism_score = (determinism_score + pattern_consistency) / 2
        
        # Check for deterministic language in outputs
        text_content = self._extract_text_from_outputs(cognitive_outputs)
        deterministic_language_score = self._assess_deterministic_language(text_content)
        
        determinism_score = (determinism_score + deterministic_language_score) / 2
        
        return min(1.0, determinism_score)
    
    def _analyze_execution_patterns(self, execution_history: List[Dict[str, Any]]) -> float:
        """Analyze execution history for consistent patterns"""
        
        if len(execution_history) < 2:
            return 0.7
        
        pattern_scores = []
        
        # Check timing consistency
        execution_times = []
        for entry in execution_history:
            if "execution_time" in entry:
                execution_times.append(entry["execution_time"])
        
        if len(execution_times) > 1:
            time_variance = statistics.variance(execution_times) if len(execution_times) > 1 else 0
            avg_time = statistics.mean(execution_times)
            if avg_time > 0:
                time_consistency = max(0.0, 1.0 - (time_variance / avg_time))
                pattern_scores.append(time_consistency)
        
        # Check protocol execution patterns
        protocol_patterns = []
        for entry in execution_history:
            if "protocols_executed" in entry:
                protocol_patterns.append(entry["protocols_executed"])
        
        if protocol_patterns:
            # Check if protocol execution is consistent
            if len(set(protocol_patterns)) == 1:  # All executions used same protocol count
                pattern_scores.append(1.0)
            else:
                pattern_variance = statistics.variance(protocol_patterns) if len(protocol_patterns) > 1 else 0
                avg_protocols = statistics.mean(protocol_patterns)
                if avg_protocols > 0:
                    protocol_consistency = max(0.0, 1.0 - (pattern_variance / avg_protocols))
                    pattern_scores.append(protocol_consistency)
        
        return sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0.7
    
    def _extract_text_from_outputs(self, cognitive_outputs: Dict[str, Any]) -> str:
        """Extract all text content from cognitive outputs"""
        
        text_parts = []
        
        def extract_text_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)
        
        extract_text_recursive(cognitive_outputs)
        return " ".join(text_parts)
    
    def _assess_deterministic_language(self, text_content: str) -> float:
        """Assess how deterministic the language used is"""
        
        if not text_content:
            return 0.7
        
        total_words = len(text_content.split())
        if total_words == 0:
            return 0.7
        
        # Count deterministic indicators
        deterministic_count = 0
        for pattern in self.deterministic_indicators:
            import re
            matches = len(re.findall(pattern, text_content, re.IGNORECASE))
            deterministic_count += matches
        
        # Count probabilistic indicators (penalty)
        probabilistic_count = 0
        for pattern in self.probabilistic_indicators:
            import re
            matches = len(re.findall(pattern, text_content, re.IGNORECASE))
            probabilistic_count += matches
        
        # Calculate determinism ratio
        deterministic_ratio = deterministic_count / total_words
        probabilistic_ratio = probabilistic_count / total_words
        
        # Higher deterministic language, lower probabilistic language = higher score
        score = 0.7 + (deterministic_ratio * 10) - (probabilistic_ratio * 5)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_execution_repeatability(self, 
                                         execution_history: List[Dict[str, Any]],
                                         input_variations: Dict[str, Any]) -> float:
        """Calculate execution repeatability score"""
        
        if not execution_history:
            return 0.8  # Default moderate repeatability
        
        repeatability_factors = []
        
        # Check for identical inputs producing identical outputs
        identical_input_groups = self._group_by_identical_inputs(execution_history, input_variations)
        
        for input_hash, executions in identical_input_groups.items():
            if len(executions) > 1:
                # Multiple executions with identical inputs
                output_consistency = self._calculate_output_consistency_within_group(executions)
                repeatability_factors.append(output_consistency)
        
        # Check for similar inputs producing similar outputs
        similar_input_groups = self._group_by_similar_inputs(execution_history, input_variations)
        
        for group in similar_input_groups:
            if len(group) > 1:
                similarity_consistency = self._calculate_similarity_consistency_within_group(group)
                repeatability_factors.append(similarity_consistency * 0.8)  # Lower weight for similar inputs
        
        if repeatability_factors:
            return sum(repeatability_factors) / len(repeatability_factors)
        else:
            return 0.8  # Default when no repeatability data available
    
    def _group_by_identical_inputs(self, 
                                 execution_history: List[Dict[str, Any]],
                                 input_variations: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group executions by identical inputs"""
        
        groups = {}
        
        for execution in execution_history:
            # Create hash of input parameters
            input_data = execution.get("input_data", {})
            input_hash = self._hash_dict(input_data)
            
            if input_hash not in groups:
                groups[input_hash] = []
            groups[input_hash].append(execution)
        
        return {k: v for k, v in groups.items() if len(v) > 1}
    
    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Create consistent hash of dictionary"""
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
            return hashlib.md5(serialized.encode()).hexdigest()
        except:
            return str(hash(str(data)))
    
    def _group_by_similar_inputs(self, 
                               execution_history: List[Dict[str, Any]],
                               input_variations: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Group executions by similar inputs"""
        
        # Simple similarity grouping - can be enhanced
        groups = []
        processed = set()
        
        for i, exec1 in enumerate(execution_history):
            if i in processed:
                continue
            
            similar_group = [exec1]
            processed.add(i)
            
            for j, exec2 in enumerate(execution_history[i+1:], i+1):
                if j in processed:
                    continue
                
                # Check input similarity
                input1 = exec1.get("input_data", {})
                input2 = exec2.get("input_data", {})
                
                similarity = self._calculate_input_similarity(input1, input2)
                if similarity > 0.8:  # High similarity threshold
                    similar_group.append(exec2)
                    processed.add(j)
            
            if len(similar_group) > 1:
                groups.append(similar_group)
        
        return groups
    
    def _calculate_input_similarity(self, input1: Dict[str, Any], input2: Dict[str, Any]) -> float:
        """Calculate similarity between two input sets"""
        return self._calculate_output_similarity(input1, input2)  # Reuse output similarity logic
    
    def _calculate_output_consistency_within_group(self, executions: List[Dict[str, Any]]) -> float:
        """Calculate output consistency within a group of executions"""
        
        if len(executions) < 2:
            return 1.0
        
        outputs = []
        for execution in executions:
            if "output" in execution:
                outputs.append(execution["output"])
        
        if len(outputs) < 2:
            return 0.8
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                similarity = self._calculate_output_similarity(outputs[i], outputs[j])
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.8
    
    def _calculate_similarity_consistency_within_group(self, executions: List[Dict[str, Any]]) -> float:
        """Calculate consistency within a group of similar executions"""
        return self._calculate_output_consistency_within_group(executions)
    
    def _calculate_response_stability(self, 
                                    cognitive_outputs: Dict[str, Any],
                                    execution_history: List[Dict[str, Any]]) -> float:
        """Calculate response stability over time"""
        
        if not execution_history:
            return 0.8  # Default stability without history
        
        stability_factors = []
        
        # Check output format stability
        current_output_structure = self._analyze_output_structure(cognitive_outputs)
        
        for execution in execution_history[-5:]:  # Last 5 executions
            if "output" in execution:
                historical_structure = self._analyze_output_structure(execution["output"])
                structure_similarity = self._compare_output_structures(
                    current_output_structure, historical_structure
                )
                stability_factors.append(structure_similarity)
        
        # Check response time stability
        response_times = []
        for execution in execution_history:
            if "execution_time" in execution:
                response_times.append(execution["execution_time"])
        
        if len(response_times) > 1:
            time_variance = statistics.variance(response_times)
            avg_time = statistics.mean(response_times)
            if avg_time > 0:
                time_stability = max(0.0, 1.0 - (time_variance / avg_time))
                stability_factors.append(time_stability)
        
        return sum(stability_factors) / len(stability_factors) if stability_factors else 0.8
    
    def _analyze_output_structure(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of output for comparison"""
        
        structure = {
            "keys": list(output.keys()) if isinstance(output, dict) else [],
            "types": {},
            "nested_levels": 0
        }
        
        if isinstance(output, dict):
            for key, value in output.items():
                structure["types"][key] = type(value).__name__
                if isinstance(value, dict):
                    structure["nested_levels"] = max(structure["nested_levels"], 1)
        
        return structure
    
    def _compare_output_structures(self, struct1: Dict[str, Any], struct2: Dict[str, Any]) -> float:
        """Compare two output structures for similarity"""
        
        similarity_factors = []
        
        # Key similarity
        keys1 = set(struct1.get("keys", []))
        keys2 = set(struct2.get("keys", []))
        
        if keys1 or keys2:
            key_similarity = len(keys1.intersection(keys2)) / len(keys1.union(keys2))
            similarity_factors.append(key_similarity)
        
        # Type similarity
        types1 = struct1.get("types", {})
        types2 = struct2.get("types", {})
        
        common_keys = set(types1.keys()).intersection(set(types2.keys()))
        if common_keys:
            type_matches = sum(1 for key in common_keys if types1[key] == types2[key])
            type_similarity = type_matches / len(common_keys)
            similarity_factors.append(type_similarity)
        
        # Structural complexity similarity
        levels1 = struct1.get("nested_levels", 0)
        levels2 = struct2.get("nested_levels", 0)
        level_similarity = 1.0 - abs(levels1 - levels2) / max(levels1, levels2, 1)
        similarity_factors.append(level_similarity)
        
        return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0
    
    def _calculate_error_predictability(self, execution_history: List[Dict[str, Any]]) -> float:
        """Calculate how predictable errors are"""
        
        error_executions = []
        successful_executions = []
        
        for execution in execution_history:
            if execution.get("status") == "error" or "error" in execution:
                error_executions.append(execution)
            else:
                successful_executions.append(execution)
        
        if not error_executions:
            return 1.0  # Perfect predictability - no errors
        
        # Analyze error patterns
        error_pattern_consistency = self._analyze_error_patterns(error_executions)
        
        # Check if errors occur under predictable conditions
        error_condition_predictability = self._assess_error_condition_predictability(
            error_executions, successful_executions
        )
        
        return (error_pattern_consistency + error_condition_predictability) / 2
    
    def _analyze_error_patterns(self, error_executions: List[Dict[str, Any]]) -> float:
        """Analyze consistency in error patterns"""
        
        if len(error_executions) < 2:
            return 1.0
        
        error_types = []
        error_messages = []
        
        for execution in error_executions:
            error_info = execution.get("error", {})
            if isinstance(error_info, dict):
                error_types.append(error_info.get("type", "unknown"))
                error_messages.append(error_info.get("message", ""))
            elif isinstance(error_info, str):
                error_messages.append(error_info)
                error_types.append("string_error")
        
        # Check error type consistency
        type_consistency = 0.0
        if error_types:
            most_common_type = max(set(error_types), key=error_types.count)
            type_consistency = error_types.count(most_common_type) / len(error_types)
        
        # Check error message similarity
        message_consistency = 0.0
        if len(error_messages) > 1:
            similarities = []
            for i in range(len(error_messages)):
                for j in range(i+1, len(error_messages)):
                    similarity = self._calculate_text_similarity(error_messages[i], error_messages[j])
                    similarities.append(similarity)
            message_consistency = sum(similarities) / len(similarities) if similarities else 0.0
        
        return (type_consistency + message_consistency) / 2
    
    def _assess_error_condition_predictability(self, 
                                             error_executions: List[Dict[str, Any]],
                                             successful_executions: List[Dict[str, Any]]) -> float:
        """Assess how predictable the conditions leading to errors are"""
        
        if not error_executions or not successful_executions:
            return 0.8  # Default moderate predictability
        
        # Analyze input conditions for errors vs successes
        error_inputs = [exec.get("input_data", {}) for exec in error_executions]
        success_inputs = [exec.get("input_data", {}) for exec in successful_executions]
        
        # Check if error inputs have distinguishable patterns
        error_patterns = self._identify_common_patterns(error_inputs)
        success_patterns = self._identify_common_patterns(success_inputs)
        
        # Calculate pattern separation
        if error_patterns and success_patterns:
            pattern_overlap = len(set(error_patterns).intersection(set(success_patterns)))
            total_patterns = len(set(error_patterns).union(set(success_patterns)))
            separation_score = 1.0 - (pattern_overlap / total_patterns) if total_patterns > 0 else 0.5
            return separation_score
        
        return 0.7  # Default moderate predictability
    
    def _identify_common_patterns(self, input_list: List[Dict[str, Any]]) -> List[str]:
        """Identify common patterns in input data"""
        
        patterns = []
        
        if not input_list:
            return patterns
        
        # Extract common keys
        all_keys = set()
        for inputs in input_list:
            if isinstance(inputs, dict):
                all_keys.update(inputs.keys())
        
        # Identify common key-value patterns
        for key in all_keys:
            values = []
            for inputs in input_list:
                if isinstance(inputs, dict) and key in inputs:
                    values.append(str(inputs[key]))
            
            if values:
                most_common_value = max(set(values), key=values.count)
                if values.count(most_common_value) > len(values) * 0.7:  # 70% threshold
                    patterns.append(f"{key}:{most_common_value}")
        
        return patterns
    
    def _count_probabilistic_variations(self, 
                                      cognitive_outputs: Dict[str, Any],
                                      execution_history: List[Dict[str, Any]]) -> int:
        """Count indicators of probabilistic variation"""
        
        variation_count = 0
        
        # Check cognitive outputs for probabilistic language
        text_content = self._extract_text_from_outputs(cognitive_outputs)
        
        for pattern in self.probabilistic_indicators:
            import re
            matches = len(re.findall(pattern, text_content, re.IGNORECASE))
            variation_count += matches
        
        # Check execution history for inconsistent patterns
        if len(execution_history) > 1:
            # Check for output format variations
            output_structures = []
            for execution in execution_history:
                if "output" in execution:
                    structure = self._analyze_output_structure(execution["output"])
                    output_structures.append(structure)
            
            if output_structures:
                unique_structures = len(set(str(s) for s in output_structures))
                if unique_structures > 1:
                    variation_count += unique_structures - 1
        
        return variation_count
    
    def _calculate_consistency_variance(self, 
                                      previous_results: List[Dict[str, Any]],
                                      cognitive_outputs: Dict[str, Any]) -> float:
        """Calculate variance in consistency measurements"""
        
        if len(previous_results) < 2:
            return 0.0  # No variance with insufficient data
        
        consistency_scores = []
        
        # Calculate consistency score for each result pair
        for i in range(len(previous_results)):
            for j in range(i+1, len(previous_results)):
                consistency = self._calculate_output_similarity(
                    previous_results[i], previous_results[j]
                )
                consistency_scores.append(consistency)
        
        # Add current output comparison
        for prev_result in previous_results:
            consistency = self._calculate_output_similarity(cognitive_outputs, prev_result)
            consistency_scores.append(consistency)
        
        if len(consistency_scores) > 1:
            return statistics.variance(consistency_scores)
        else:
            return 0.0
    
    def _calculate_deterministic_behavior_ratio(self, 
                                              cognitive_outputs: Dict[str, Any],
                                              execution_history: List[Dict[str, Any]]) -> float:
        """Calculate ratio of deterministic to probabilistic behavior"""
        
        text_content = self._extract_text_from_outputs(cognitive_outputs)
        
        # Add text from execution history
        for execution in execution_history:
            if "output" in execution:
                hist_text = self._extract_text_from_outputs(execution["output"])
                text_content += " " + hist_text
        
        if not text_content:
            return 0.7  # Default moderate determinism
        
        total_words = len(text_content.split())
        if total_words == 0:
            return 0.7
        
        # Count deterministic vs probabilistic indicators
        deterministic_count = 0
        probabilistic_count = 0
        
        for pattern in self.deterministic_indicators:
            import re
            matches = len(re.findall(pattern, text_content, re.IGNORECASE))
            deterministic_count += matches
        
        for pattern in self.probabilistic_indicators:
            import re
            matches = len(re.findall(pattern, text_content, re.IGNORECASE))
            probabilistic_count += matches
        
        total_indicators = deterministic_count + probabilistic_count
        
        if total_indicators == 0:
            return 0.7  # Neutral when no indicators found
        
        return deterministic_count / total_indicators
    
    def _calculate_reliability_confidence(self, 
                                        output_consistency: float,
                                        behavioral_determinism: float,
                                        execution_repeatability: float) -> float:
        """Calculate confidence in reliability measurements"""
        
        # Higher consistency and determinism = higher confidence
        core_factors = [output_consistency, behavioral_determinism, execution_repeatability]
        average_reliability = sum(core_factors) / len(core_factors)
        
        # Confidence increases with higher and more consistent reliability metrics
        variance = statistics.variance(core_factors) if len(core_factors) > 1 else 0
        confidence = average_reliability * (1.0 - variance)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_overall_predictability(self, 
                                        output_consistency: float,
                                        behavioral_determinism: float,
                                        response_stability: float,
                                        error_predictability: float) -> float:
        """Calculate overall predictability score"""
        
        weights = {
            "output_consistency": 0.3,
            "behavioral_determinism": 0.25,
            "response_stability": 0.25,
            "error_predictability": 0.2
        }
        
        weighted_score = (
            output_consistency * weights["output_consistency"] +
            behavioral_determinism * weights["behavioral_determinism"] +
            response_stability * weights["response_stability"] +
            error_predictability * weights["error_predictability"]
        )
        
        return weighted_score
    
    def _analyze_output_consistency(self, 
                                  cognitive_outputs: Dict[str, Any],
                                  previous_results: List[Dict[str, Any]],
                                  input_variations: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive output consistency analysis"""
        
        analysis = {
            "consistency_level": ConsistencyLevel.MODERATE,
            "consistency_score": 0.0,
            "format_consistency": 0.0,
            "content_consistency": 0.0,
            "structural_consistency": 0.0,
            "inconsistencies": []
        }
        
        # Calculate consistency score
        analysis["consistency_score"] = self._calculate_output_consistency(
            cognitive_outputs, previous_results
        )
        
        # Determine consistency level
        if analysis["consistency_score"] >= 0.95:
            analysis["consistency_level"] = ConsistencyLevel.PERFECT
        elif analysis["consistency_score"] >= 0.9:
            analysis["consistency_level"] = ConsistencyLevel.EXCELLENT
        elif analysis["consistency_score"] >= 0.8:
            analysis["consistency_level"] = ConsistencyLevel.GOOD
        elif analysis["consistency_score"] >= 0.6:
            analysis["consistency_level"] = ConsistencyLevel.MODERATE
        elif analysis["consistency_score"] >= 0.4:
            analysis["consistency_level"] = ConsistencyLevel.POOR
        else:
            analysis["consistency_level"] = ConsistencyLevel.UNRELIABLE
        
        # Analyze specific aspects of consistency
        if previous_results:
            format_scores = []
            content_scores = []
            structural_scores = []
            
            current_structure = self._analyze_output_structure(cognitive_outputs)
            
            for prev_result in previous_results[-3:]:  # Last 3 for detailed analysis
                if isinstance(prev_result, dict):
                    prev_structure = self._analyze_output_structure(prev_result)
                    
                    # Format consistency
                    format_similarity = self._compare_output_structures(current_structure, prev_structure)
                    format_scores.append(format_similarity)
                    
                    # Content consistency
                    content_similarity = self._calculate_output_similarity(cognitive_outputs, prev_result)
                    content_scores.append(content_similarity)
                    
                    # Structural consistency
                    structural_scores.append(format_similarity)  # Same as format for now
            
            analysis["format_consistency"] = sum(format_scores) / len(format_scores) if format_scores else 0.8
            analysis["content_consistency"] = sum(content_scores) / len(content_scores) if content_scores else 0.8
            analysis["structural_consistency"] = sum(structural_scores) / len(structural_scores) if structural_scores else 0.8
            
            # Identify inconsistencies
            if analysis["format_consistency"] < 0.7:
                analysis["inconsistencies"].append("format_variations")
            if analysis["content_consistency"] < 0.7:
                analysis["inconsistencies"].append("content_variations")
            if analysis["structural_consistency"] < 0.7:
                analysis["inconsistencies"].append("structural_variations")
        
        return analysis
    
    def _assess_behavioral_determinism(self, 
                                     execution_history: List[Dict[str, Any]],
                                     protocol_stack: List[str],
                                     cognitive_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive behavioral determinism assessment"""
        
        assessment = {
            "determinism_score": 0.0,
            "execution_pattern_consistency": 0.0,
            "language_determinism": 0.0,
            "protocol_determinism": 0.0,
            "deterministic_indicators": [],
            "probabilistic_indicators": []
        }
        
        # Calculate determinism score
        assessment["determinism_score"] = self._calculate_behavioral_determinism(
            execution_history, cognitive_outputs
        )
        
        # Analyze execution patterns
        if execution_history:
            assessment["execution_pattern_consistency"] = self._analyze_execution_patterns(execution_history)
        else:
            assessment["execution_pattern_consistency"] = 0.8
        
        # Analyze language determinism
        text_content = self._extract_text_from_outputs(cognitive_outputs)
        assessment["language_determinism"] = self._assess_deterministic_language(text_content)
        
        # Assess protocol determinism
        assessment["protocol_determinism"] = self._assess_protocol_determinism(protocol_stack)
        
        # Identify specific indicators
        for pattern in self.deterministic_indicators:
            import re
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                assessment["deterministic_indicators"].extend(matches)
        
        for pattern in self.probabilistic_indicators:
            import re
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                assessment["probabilistic_indicators"].extend(matches)
        
        return assessment
    
    def _assess_protocol_determinism(self, protocol_stack: List[str]) -> float:
        """Assess how deterministic the protocol stack is"""
        
        if not protocol_stack:
            return 0.7
        
        deterministic_protocol_indicators = [
            "Deterministic", "Validation", "Verification", "Consistency",
            "Truth", "Reliability", "Quality", "Governance"
        ]
        
        probabilistic_protocol_indicators = [
            "Random", "Probabilistic", "Estimate", "Approximate", "Variable"
        ]
        
        deterministic_count = 0
        probabilistic_count = 0
        
        for protocol in protocol_stack:
            for indicator in deterministic_protocol_indicators:
                if indicator in protocol:
                    deterministic_count += 1
                    break
            
            for indicator in probabilistic_protocol_indicators:
                if indicator in protocol:
                    probabilistic_count += 1
                    break
        
        total_relevant_protocols = deterministic_count + probabilistic_count
        
        if total_relevant_protocols == 0:
            return 0.7  # Neutral when no clear indicators
        
        return deterministic_count / total_relevant_protocols
    
    def _check_execution_repeatability(self, 
                                     execution_history: List[Dict[str, Any]],
                                     previous_results: List[Dict[str, Any]],
                                     input_variations: Dict[str, Any]) -> Dict[str, Any]:
        """Check execution repeatability comprehensively"""
        
        repeatability_check = {
            "repeatability_score": 0.0,
            "identical_input_consistency": 0.0,
            "similar_input_consistency": 0.0,
            "execution_stability": 0.0,
            "repeatability_violations": []
        }
        
        # Calculate repeatability score
        repeatability_check["repeatability_score"] = self._calculate_execution_repeatability(
            execution_history, input_variations
        )
        
        # Check identical input consistency
        identical_groups = self._group_by_identical_inputs(execution_history, input_variations)
        if identical_groups:
            identical_scores = []
            for group in identical_groups.values():
                consistency = self._calculate_output_consistency_within_group(group)
                identical_scores.append(consistency)
                
                if consistency < 0.9:  # High threshold for identical inputs
                    repeatability_check["repeatability_violations"].append(
                        f"Identical inputs produced inconsistent outputs (consistency: {consistency:.3f})"
                    )
            
            repeatability_check["identical_input_consistency"] = sum(identical_scores) / len(identical_scores)
        else:
            repeatability_check["identical_input_consistency"] = 0.8  # No data available
        
        # Check similar input consistency
        similar_groups = self._group_by_similar_inputs(execution_history, input_variations)
        if similar_groups:
            similar_scores = []
            for group in similar_groups:
                consistency = self._calculate_similarity_consistency_within_group(group)
                similar_scores.append(consistency)
                
                if consistency < 0.7:  # Lower threshold for similar inputs
                    repeatability_check["repeatability_violations"].append(
                        f"Similar inputs produced highly inconsistent outputs (consistency: {consistency:.3f})"
                    )
            
            repeatability_check["similar_input_consistency"] = sum(similar_scores) / len(similar_scores)
        else:
            repeatability_check["similar_input_consistency"] = 0.8
        
        # Assess execution stability
        if execution_history:
            execution_times = [exec.get("execution_time", 1.0) for exec in execution_history]
            if len(execution_times) > 1:
                time_variance = statistics.variance(execution_times)
                avg_time = statistics.mean(execution_times)
                if avg_time > 0:
                    repeatability_check["execution_stability"] = max(0.0, 1.0 - (time_variance / avg_time))
                else:
                    repeatability_check["execution_stability"] = 1.0
            else:
                repeatability_check["execution_stability"] = 1.0
        
        return repeatability_check
    
    def _evaluate_response_stability(self, 
                                   cognitive_outputs: Dict[str, Any],
                                   execution_history: List[Dict[str, Any]],
                                   execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate response stability comprehensively"""
        
        stability_evaluation = {
            "stability_score": 0.0,
            "format_stability": 0.0,
            "timing_stability": 0.0,
            "content_stability": 0.0,
            "stability_issues": []
        }
        
        # Calculate stability score
        stability_evaluation["stability_score"] = self._calculate_response_stability(
            cognitive_outputs, execution_history
        )
        
        # Analyze format stability
        current_structure = self._analyze_output_structure(cognitive_outputs)
        if execution_history:
            format_similarities = []
            for execution in execution_history[-5:]:
                if "output" in execution:
                    hist_structure = self._analyze_output_structure(execution["output"])
                    similarity = self._compare_output_structures(current_structure, hist_structure)
                    format_similarities.append(similarity)
            
            stability_evaluation["format_stability"] = sum(format_similarities) / len(format_similarities) if format_similarities else 0.8
        else:
            stability_evaluation["format_stability"] = 0.8
        
        # Analyze timing stability
        current_time = execution_metrics.get("execution_time", 1.0)
        if execution_history:
            historical_times = [exec.get("execution_time", 1.0) for exec in execution_history]
            all_times = historical_times + [current_time]
            
            if len(all_times) > 1:
                time_variance = statistics.variance(all_times)
                avg_time = statistics.mean(all_times)
                if avg_time > 0:
                    stability_evaluation["timing_stability"] = max(0.0, 1.0 - (time_variance / avg_time))
                else:
                    stability_evaluation["timing_stability"] = 1.0
            else:
                stability_evaluation["timing_stability"] = 1.0
        else:
            stability_evaluation["timing_stability"] = 1.0
        
        # Analyze content stability
        if execution_history:
            content_similarities = []
            for execution in execution_history[-3:]:  # Recent executions
                if "output" in execution:
                    similarity = self._calculate_output_similarity(cognitive_outputs, execution["output"])
                    content_similarities.append(similarity)
            
            stability_evaluation["content_stability"] = sum(content_similarities) / len(content_similarities) if content_similarities else 0.8
        else:
            stability_evaluation["content_stability"] = 0.8
        
        # Identify stability issues
        if stability_evaluation["format_stability"] < 0.7:
            stability_evaluation["stability_issues"].append("format_instability")
        if stability_evaluation["timing_stability"] < 0.7:
            stability_evaluation["stability_issues"].append("timing_instability")
        if stability_evaluation["content_stability"] < 0.7:
            stability_evaluation["stability_issues"].append("content_instability")
        
        return stability_evaluation
    
    def _detect_probabilistic_variations(self, 
                                       cognitive_outputs: Dict[str, Any],
                                       execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect probabilistic variations comprehensively"""
        
        detection = {
            "variations_detected": False,
            "variation_count": 0,
            "probabilistic_language_ratio": 0.0,
            "variation_examples": [],
            "variation_severity": "none"
        }
        
        # Count probabilistic variations
        detection["variation_count"] = self._count_probabilistic_variations(
            cognitive_outputs, execution_history
        )
        detection["variations_detected"] = detection["variation_count"] > 0
        
        # Calculate probabilistic language ratio
        text_content = self._extract_text_from_outputs(cognitive_outputs)
        total_words = len(text_content.split())
        
        if total_words > 0:
            probabilistic_words = 0
            for pattern in self.probabilistic_indicators:
                import re
                matches = len(re.findall(pattern, text_content, re.IGNORECASE))
                probabilistic_words += matches
            
            detection["probabilistic_language_ratio"] = probabilistic_words / total_words
        
        # Collect variation examples
        for pattern in self.probabilistic_indicators:
            import re
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            detection["variation_examples"].extend(matches)
        
        # Determine severity
        if detection["variation_count"] > 10 or detection["probabilistic_language_ratio"] > 0.3:
            detection["variation_severity"] = "high"
        elif detection["variation_count"] > 5 or detection["probabilistic_language_ratio"] > 0.2:
            detection["variation_severity"] = "medium"
        elif detection["variation_count"] > 0 or detection["probabilistic_language_ratio"] > 0.1:
            detection["variation_severity"] = "low"
        
        return detection
    
    def _assess_error_predictability(self, 
                                   execution_history: List[Dict[str, Any]],
                                   execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess error predictability comprehensively"""
        
        assessment = {
            "predictability_score": 0.0,
            "error_pattern_consistency": 0.0,
            "error_condition_predictability": 0.0,
            "error_count": 0,
            "success_count": 0,
            "predictable_errors": []
        }
        
        # Calculate predictability score
        assessment["predictability_score"] = self._calculate_error_predictability(execution_history)
        
        # Separate errors and successes
        error_executions = []
        successful_executions = []
        
        for execution in execution_history:
            if execution.get("status") == "error" or "error" in execution:
                error_executions.append(execution)
            else:
                successful_executions.append(execution)
        
        assessment["error_count"] = len(error_executions)
        assessment["success_count"] = len(successful_executions)
        
        # Analyze error patterns
        if error_executions:
            assessment["error_pattern_consistency"] = self._analyze_error_patterns(error_executions)
        else:
            assessment["error_pattern_consistency"] = 1.0  # No errors = perfect predictability
        
        # Analyze error conditions
        if error_executions and successful_executions:
            assessment["error_condition_predictability"] = self._assess_error_condition_predictability(
                error_executions, successful_executions
            )
        else:
            assessment["error_condition_predictability"] = 0.8
        
        # Identify predictable error types
        for execution in error_executions:
            error_info = execution.get("error", {})
            if isinstance(error_info, dict):
                error_type = error_info.get("type", "unknown")
                if error_type in self.error_patterns:
                    assessment["predictable_errors"].append(error_type)
        
        return assessment
    
    def _calculate_compliance_score(self, reliability_metrics: ReliabilityMetrics,
                                  consistency_analysis: Dict[str, Any],
                                  determinism_assessment: Dict[str, Any],
                                  probabilistic_detection: Dict[str, Any]) -> float:
        """Calculate overall compliance score for Law 5"""
        
        # Core reliability metrics (60% weight)
        core_score = (
            reliability_metrics.output_consistency_score * 0.20 +
            reliability_metrics.behavioral_determinism_score * 0.15 +
            reliability_metrics.execution_repeatability_score * 0.15 +
            reliability_metrics.response_stability_score * 0.10
        )
        
        # Predictability assessment (25% weight)
        predictability_score = reliability_metrics.overall_predictability_score * 0.25
        
        # Error predictability (15% weight)
        error_score = reliability_metrics.error_predictability_score * 0.15
        
        # Penalties for probabilistic variations
        probabilistic_penalty = 0.0
        if probabilistic_detection["variations_detected"]:
            severity_penalties = {"low": 0.05, "medium": 0.10, "high": 0.20}
            probabilistic_penalty = severity_penalties.get(probabilistic_detection["variation_severity"], 0.0)
        
        total_score = core_score + predictability_score + error_score - probabilistic_penalty
        
        return max(0.0, min(1.0, total_score))
    
    def _identify_violations(self, reliability_metrics: ReliabilityMetrics,
                           probabilistic_detection: Dict[str, Any],
                           consistency_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific violations of Law 5"""
        
        violations = []
        
        # Check output consistency
        if reliability_metrics.output_consistency_score < self.reliability_requirements["minimum_output_consistency"]:
            violations.append({
                "type": "insufficient_output_consistency",
                "severity": "high",
                "description": f"Output consistency {reliability_metrics.output_consistency_score:.3f} below required {self.reliability_requirements['minimum_output_consistency']}",
                "law": "Law5_DeterministicReliability",
                "remediation": "Implement deterministic output generation mechanisms"
            })
        
        # Check behavioral determinism
        if reliability_metrics.behavioral_determinism_score < self.reliability_requirements["minimum_behavioral_determinism"]:
            violations.append({
                "type": "insufficient_behavioral_determinism",
                "severity": "high",
                "description": f"Behavioral determinism {reliability_metrics.behavioral_determinism_score:.3f} below required {self.reliability_requirements['minimum_behavioral_determinism']}",
                "law": "Law5_DeterministicReliability",
                "remediation": "Replace probabilistic behavior with deterministic algorithms"
            })
        
        # Check execution repeatability
        if reliability_metrics.execution_repeatability_score < self.reliability_requirements["minimum_repeatability"]:
            violations.append({
                "type": "insufficient_execution_repeatability",
                "severity": "medium",
                "description": f"Execution repeatability {reliability_metrics.execution_repeatability_score:.3f} below required {self.reliability_requirements['minimum_repeatability']}",
                "law": "Law5_DeterministicReliability",
                "remediation": "Ensure identical inputs produce identical outputs consistently"
            })
        
        # Check probabilistic variations
        if reliability_metrics.probabilistic_variation_count > 5:  # Threshold for concern
            violations.append({
                "type": "excessive_probabilistic_variations",
                "severity": probabilistic_detection["variation_severity"],
                "description": f"Probabilistic variation count {reliability_metrics.probabilistic_variation_count} indicates non-deterministic behavior",
                "law": "Law5_DeterministicReliability",
                "remediation": "Replace probabilistic language and behavior with deterministic alternatives"
            })
        
        # Check response stability
        if reliability_metrics.response_stability_score < self.reliability_requirements["minimum_response_stability"]:
            violations.append({
                "type": "insufficient_response_stability",
                "severity": "medium",
                "description": f"Response stability {reliability_metrics.response_stability_score:.3f} below required {self.reliability_requirements['minimum_response_stability']}",
                "law": "Law5_DeterministicReliability",
                "remediation": "Stabilize response formats and timing for consistent behavior"
            })
        
        # Check consistency level
        if consistency_analysis["consistency_level"] in [ConsistencyLevel.POOR, ConsistencyLevel.UNRELIABLE]:
            violations.append({
                "type": "poor_consistency_level",
                "severity": "high" if consistency_analysis["consistency_level"] == ConsistencyLevel.UNRELIABLE else "medium",
                "description": f"Consistency level is {consistency_analysis['consistency_level'].value}, indicating unreliable behavior",
                "law": "Law5_DeterministicReliability",
                "remediation": "Implement comprehensive consistency enforcement mechanisms"
            })
        
        return violations
    
    def _generate_recommendations(self, reliability_metrics: ReliabilityMetrics, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving deterministic reliability"""
        
        recommendations = []
        
        # Performance-based recommendations
        if reliability_metrics.output_consistency_score < 0.9:
            recommendations.append("Implement deterministic output generation with consistent formatting")
        
        if reliability_metrics.behavioral_determinism_score < 0.85:
            recommendations.append("Replace probabilistic algorithms with deterministic alternatives")
        
        if reliability_metrics.execution_repeatability_score < 0.95:
            recommendations.append("Ensure identical inputs always produce identical outputs")
        
        if reliability_metrics.response_stability_score < 0.85:
            recommendations.append("Stabilize response timing and format consistency")
        
        if reliability_metrics.probabilistic_variation_count > 3:
            recommendations.append("Eliminate probabilistic language and uncertainty expressions")
        
        # Violation-based recommendations
        for violation in violations:
            if "remediation" in violation and violation["remediation"] not in recommendations:
                recommendations.append(violation["remediation"])
        
        # Specific technical recommendations
        if reliability_metrics.consistency_variance > 0.1:
            recommendations.append("Reduce output variance through stricter consistency controls")
        
        if reliability_metrics.deterministic_behavior_ratio < 0.8:
            recommendations.append("Increase deterministic behavior patterns in cognitive processes")
        
        # General best practices
        if len(recommendations) == 0:
            recommendations.append("Deterministic reliability is well-implemented - maintain current consistency standards")
        
        return recommendations
    
    def _metrics_to_dict(self, metrics: ReliabilityMetrics) -> Dict[str, Any]:
        """Convert ReliabilityMetrics to dictionary for JSON serialization"""
        return {
            "output_consistency_score": metrics.output_consistency_score,
            "behavioral_determinism_score": metrics.behavioral_determinism_score,
            "execution_repeatability_score": metrics.execution_repeatability_score,
            "response_stability_score": metrics.response_stability_score,
            "error_predictability_score": metrics.error_predictability_score,
            "probabilistic_variation_count": metrics.probabilistic_variation_count,
            "consistency_variance": metrics.consistency_variance,
            "deterministic_behavior_ratio": metrics.deterministic_behavior_ratio,
            "reliability_confidence": metrics.reliability_confidence,
            "overall_predictability_score": metrics.overall_predictability_score
        }


# Test and example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_deterministic_reliability_protocol():
        protocol = DeterministicReliabilityProtocol()
        
        test_data = {
            "cognitive_outputs": {
                "analysis_result": "The system consistently produces reliable outputs based on established protocols. The analysis definitively shows clear patterns.",
                "conclusion": {
                    "certainty": "definite",
                    "reliability_score": 0.94,
                    "consistency_check": True
                }
            },
            "execution_history": [
                {
                    "input_data": {"query": "test_query", "parameters": {"mode": "analysis"}},
                    "output": {"result": "consistent_analysis", "score": 0.92},
                    "execution_time": 1.2,
                    "status": "success"
                },
                {
                    "input_data": {"query": "test_query", "parameters": {"mode": "analysis"}},
                    "output": {"result": "consistent_analysis", "score": 0.91},
                    "execution_time": 1.15,
                    "status": "success"
                }
            ],
            "previous_results": [
                {"analysis_result": "The system consistently produces reliable outputs", "score": 0.93},
                {"analysis_result": "The system produces reliable outputs consistently", "score": 0.92}
            ],
            "protocol_stack": [
                "DeterministicReliabilityProtocol", "ValidationProtocol", 
                "ConsistencyProtocol", "QualityAssuranceProtocol"
            ],
            "execution_metrics": {
                "execution_time": 1.18,
                "consistency_score": 0.96
            },
            "input_variations": {
                "identical_inputs": 2,
                "similar_inputs": 3
            }
        }
        
        result = await protocol.execute(test_data)
        print("Deterministic Reliability Protocol Test Results:")
        print(f"Compliance Score: {result['compliance_score']:.3f}")
        print(f"Status: {result['status']}")
        print(f"Output Consistency: {result['reliability_metrics']['output_consistency_score']:.3f}")
        print(f"Behavioral Determinism: {result['reliability_metrics']['behavioral_determinism_score']:.3f}")
        print(f"Execution Repeatability: {result['reliability_metrics']['execution_repeatability_score']:.3f}")
        print(f"Probabilistic Variations: {result['reliability_metrics']['probabilistic_variation_count']}")
        print(f"Violations: {len(result['violations'])}")
        
        for recommendation in result['recommendations']:
            print(f"- {recommendation}")
    
    asyncio.run(test_deterministic_reliability_protocol())