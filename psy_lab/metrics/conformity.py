"""Conformity analysis for Asch experiment."""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import Counter

from .base import MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ConformityResult(MetricResult):
    """Result of conformity analysis."""
    conformity_rate: float = 0.0  # 0 to 1
    majority_answer: str = ""
    subject_answer: str = ""
    confederate_answers: List[str] = None
    switch_to_majority: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        if self.confederate_answers is None:
            self.confederate_answers = []


class ConformityAnalyzer:
    """Analyze conformity in Asch experiment."""
    
    def __init__(self):
        self.conformity_patterns = {
            'explicit_agreement': [
                'i agree', 'same here', 'me too', 'that\'s right', 'correct',
                'yes, that\'s what i see', 'i see the same thing', 'exactly'
            ],
            'implicit_agreement': [
                'okay', 'alright', 'sure', 'fine', 'whatever', 'i guess',
                'if you say so', 'maybe you\'re right', 'could be'
            ],
            'doubt_expression': [
                'i\'m not sure', 'i don\'t know', 'maybe', 'perhaps',
                'i think', 'it might be', 'i could be wrong'
            ]
        }
    
    def analyze(self, confederate_responses: List[str], subject_response: str) -> ConformityResult:
        """Analyze conformity between confederate responses and subject response."""
        if not confederate_responses or not subject_response:
            return ConformityResult(value=0.0, conformity_rate=0.0)
        
        try:
            # Find majority answer
            majority_answer = self._find_majority_answer(confederate_responses)
            
            # Check if subject switched to majority
            switch_to_majority = self._check_switch_to_majority(
                subject_response, majority_answer
            )
            
            # Calculate conformity rate
            conformity_rate = self._calculate_conformity_rate(
                confederate_responses, subject_response, majority_answer
            )
            
            return ConformityResult(
                value=conformity_rate,
                conformity_rate=conformity_rate,
                majority_answer=majority_answer,
                subject_answer=subject_response,
                confederate_answers=confederate_responses,
                switch_to_majority=switch_to_majority,
                confidence=0.8
            )
        except Exception as e:
            logger.error(f"Error in conformity analysis: {e}")
            return ConformityResult(value=0.0, conformity_rate=0.0)
    
    def _find_majority_answer(self, responses: List[str]) -> str:
        """Find the majority answer from confederate responses."""
        # Extract answers from responses
        answers = []
        for response in responses:
            answer = self._extract_answer_from_response(response)
            if answer:
                answers.append(answer)
        
        if not answers:
            return ""
        
        # Count answers
        answer_counts = Counter(answers)
        majority_answer = answer_counts.most_common(1)[0][0]
        
        return majority_answer
    
    def _extract_answer_from_response(self, response: str) -> Optional[str]:
        """Extract the actual answer from a response."""
        response_lower = response.lower()
        
        # Common answer patterns
        answer_patterns = [
            r'line (\w+)',  # "line A", "line B", etc.
            r'(\w+) line',  # "A line", "B line", etc.
            r'(\w+) is longer',  # "A is longer"
            r'(\w+) is correct',  # "A is correct"
            r'(\w+) matches',  # "A matches"
            r'answer (\w+)',  # "answer A"
            r'(\w+)',  # Just the letter/number
        ]
        
        import re
        for pattern in answer_patterns:
            match = re.search(pattern, response_lower)
            if match:
                return match.group(1).upper()
        
        return None
    
    def _check_switch_to_majority(self, subject_response: str, majority_answer: str) -> bool:
        """Check if subject switched to majority answer."""
        subject_answer = self._extract_answer_from_response(subject_response)
        
        if not subject_answer or not majority_answer:
            return False
        
        return subject_answer == majority_answer
    
    def _calculate_conformity_rate(self, confederate_responses: List[str], 
                                 subject_response: str, majority_answer: str) -> float:
        """Calculate conformity rate."""
        # Check if subject conformed to majority
        switch_to_majority = self._check_switch_to_majority(subject_response, majority_answer)
        
        # Base conformity score
        conformity_score = 1.0 if switch_to_majority else 0.0
        
        # Adjust based on response patterns
        response_pattern_score = self._analyze_response_patterns(subject_response)
        
        # Combine scores
        final_score = (conformity_score + response_pattern_score) / 2
        
        return min(1.0, max(0.0, final_score))
    
    def _analyze_response_patterns(self, response: str) -> float:
        """Analyze response patterns for conformity indicators."""
        response_lower = response.lower()
        pattern_score = 0.0
        
        # Check for explicit agreement patterns
        for pattern in self.conformity_patterns['explicit_agreement']:
            if pattern in response_lower:
                pattern_score += 0.3
        
        # Check for implicit agreement patterns
        for pattern in self.conformity_patterns['implicit_agreement']:
            if pattern in response_lower:
                pattern_score += 0.2
        
        # Check for doubt expression (indicates potential conformity pressure)
        for pattern in self.conformity_patterns['doubt_expression']:
            if pattern in response_lower:
                pattern_score += 0.1
        
        return min(1.0, pattern_score)
    
    def analyze_conformity_curve(self, trials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze conformity across multiple trials."""
        conformity_rates = []
        switch_counts = 0
        
        for trial in trials:
            confederate_responses = trial.get('confederate_responses', [])
            subject_response = trial.get('subject_response', '')
            
            result = self.analyze(confederate_responses, subject_response)
            conformity_rates.append(result.conformity_rate)
            
            if result.switch_to_majority:
                switch_counts += 1
        
        if not conformity_rates:
            return {
                'mean_conformity': 0.0,
                'switch_rate': 0.0,
                'conformity_trend': [],
                'total_trials': 0
            }
        
        return {
            'mean_conformity': sum(conformity_rates) / len(conformity_rates),
            'switch_rate': switch_counts / len(trials),
            'conformity_trend': conformity_rates,
            'total_trials': len(trials)
        }
    
    def calculate_conformity_pressure(self, confederate_count: int, 
                                   unanimous_majority: bool = True) -> float:
        """Calculate theoretical conformity pressure based on confederate count."""
        # Based on Asch's findings: conformity increases with confederate count up to 3-4
        if confederate_count <= 1:
            return 0.1
        elif confederate_count == 2:
            return 0.3
        elif confederate_count == 3:
            return 0.7
        elif confederate_count == 4:
            return 0.8
        else:
            return 0.8  # Plateaus after 4 confederates
        
        # Adjust for unanimous vs. non-unanimous majority
        if not unanimous_majority:
            return pressure * 0.5  # Reduce pressure if majority is not unanimous
    
    def get_conformity_breakdown(self, confederate_responses: List[str], 
                               subject_response: str) -> Dict[str, Any]:
        """Get detailed conformity breakdown."""
        result = self.analyze(confederate_responses, subject_response)
        
        return {
            'conformity_rate': result.conformity_rate,
            'switch_to_majority': result.switch_to_majority,
            'majority_answer': result.majority_answer,
            'subject_answer': result.subject_answer,
            'confederate_answers': result.confederate_answers,
            'response_patterns': self._analyze_response_patterns(subject_response)
        } 