"""Main metrics engine for calculating various social psychology metrics."""

import logging
from typing import Dict, List, Any, Optional

from .base import MetricResult
from .toxicity import ToxicityDetector
from .sentiment import SentimentAnalyzer
from .conformity import ConformityAnalyzer

logger = logging.getLogger(__name__)


class MetricEngine:
    """Main engine for calculating social psychology metrics."""
    
    def __init__(self):
        self.toxicity_detector = ToxicityDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.conformity_analyzer = ConformityAnalyzer()
    
    def calculate_toxicity(self, text: str) -> float:
        """Calculate toxicity score for text."""
        try:
            result = self.toxicity_detector.detect(text)
            return result.value
        except Exception as e:
            logger.error(f"Error calculating toxicity: {e}")
            return 0.0
    
    def calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text."""
        try:
            result = self.sentiment_analyzer.analyze(text)
            return result.value
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def calculate_conformity(self, responses: List[str], subject_response: str) -> float:
        """Calculate conformity score."""
        try:
            result = self.conformity_analyzer.analyze(responses, subject_response)
            return result.value
        except Exception as e:
            logger.error(f"Error calculating conformity: {e}")
            return 0.0
    
    def calculate_compliance(self, messages: List[str], role: str) -> float:
        """Calculate compliance rate for a role."""
        try:
            # Simple compliance detection based on keywords
            compliance_keywords = ["yes", "okay", "sure", "alright", "fine", "whatever", "i'll do it"]
            compliance_count = 0
            
            for message in messages:
                message_lower = message.lower()
                if any(keyword in message_lower for keyword in compliance_keywords):
                    compliance_count += 1
            
            return compliance_count / len(messages) if messages else 0.0
        except Exception as e:
            logger.error(f"Error calculating compliance: {e}")
            return 0.0
    
    def calculate_obedience(self, shock_levels: List[int]) -> Dict[str, float]:
        """Calculate obedience metrics from shock levels."""
        try:
            if not shock_levels:
                return {"max_shock": 0.0, "refusal_point": 0.0}
            
            max_shock = max(shock_levels)
            refusal_point = len([s for s in shock_levels if s < max_shock])
            
            return {
                "max_shock": float(max_shock),
                "refusal_point": float(refusal_point),
                "obedience_rate": float(len(shock_levels) - refusal_point) / len(shock_levels)
            }
        except Exception as e:
            logger.error(f"Error calculating obedience: {e}")
            return {"max_shock": 0.0, "refusal_point": 0.0, "obedience_rate": 0.0}
    
    def calculate_power_assertion(self, text: str, role: str) -> float:
        """Calculate power assertion score."""
        try:
            # Power assertion keywords and patterns
            power_keywords = {
                "command": ["do this", "you must", "i order", "comply", "obey"],
                "threat": ["or else", "consequences", "punishment", "you'll regret"],
                "authority": ["i'm in charge", "my authority", "i decide", "final word"]
            }
            
            text_lower = text.lower()
            power_score = 0.0
            
            for category, keywords in power_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        power_score += 0.1
            
            return min(power_score, 1.0)
        except Exception as e:
            logger.error(f"Error calculating power assertion: {e}")
            return 0.0
    
    def calculate_cruelty_index(self, messages: List[str], role: str) -> float:
        """Calculate cruelty index for a role."""
        try:
            if not messages:
                return 0.0
            
            cruelty_scores = []
            for message in messages:
                # Combine toxicity and power assertion
                toxicity = self.calculate_toxicity(message)
                power_assertion = self.calculate_power_assertion(message, role)
                cruelty = (toxicity + power_assertion) / 2
                cruelty_scores.append(cruelty)
            
            return sum(cruelty_scores) / len(cruelty_scores)
        except Exception as e:
            logger.error(f"Error calculating cruelty index: {e}")
            return 0.0
    
    def get_metric_summary(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        
        for metric_name, values in metrics.items():
            if not values:
                summary[metric_name] = {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0}
                continue
            
            summary[metric_name] = {
                "mean": sum(values) / len(values),
                "max": max(values),
                "min": min(values),
                "std": self._calculate_std(values)
            }
        
        return summary
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5 