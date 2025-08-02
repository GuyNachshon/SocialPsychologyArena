"""Toxicity detection using detoxify."""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    from detoxify import Detoxify
except ImportError:
    Detoxify = None

from .base import MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ToxicityResult(MetricResult):
    """Result of toxicity detection."""
    categories: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.categories is None:
            self.categories = {}


class ToxicityDetector:
    """Detect toxicity in text using detoxify."""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the detoxify model."""
        try:
            if Detoxify is not None:
                self.model = Detoxify('original')
                logger.info("Loaded detoxify model successfully")
            else:
                logger.warning("Detoxify not available, using fallback method")
                self.model = None
        except Exception as e:
            logger.error(f"Failed to load detoxify model: {e}")
            self.model = None
    
    def detect(self, text: str) -> ToxicityResult:
        """Detect toxicity in text."""
        if not text or not text.strip():
            return ToxicityResult(value=0.0, categories={})
        
        try:
            if self.model is not None:
                return self._detect_with_detoxify(text)
            else:
                return self._detect_fallback(text)
        except Exception as e:
            logger.error(f"Error in toxicity detection: {e}")
            return ToxicityResult(value=0.0, categories={})
    
    def _detect_with_detoxify(self, text: str) -> ToxicityResult:
        """Detect toxicity using detoxify."""
        results = self.model.predict(text)
        
        # Extract toxicity categories
        categories = {
            'toxicity': results.get('toxicity', 0.0),
            'severe_toxicity': results.get('severe_toxicity', 0.0),
            'obscene': results.get('obscene', 0.0),
            'threat': results.get('threat', 0.0),
            'insult': results.get('insult', 0.0),
            'identity_attack': results.get('identity_attack', 0.0)
        }
        
        # Calculate overall toxicity score (weighted average)
        weights = {
            'toxicity': 0.3,
            'severe_toxicity': 0.4,
            'obscene': 0.1,
            'threat': 0.2,
            'insult': 0.1,
            'identity_attack': 0.2
        }
        
        overall_score = sum(
            categories[cat] * weights[cat] 
            for cat in weights.keys() 
            if cat in categories
        )
        
        return ToxicityResult(
            value=overall_score,
            categories=categories,
            confidence=0.8  # Detoxify confidence
        )
    
    def _detect_fallback(self, text: str) -> ToxicityResult:
        """Fallback toxicity detection using keyword matching."""
        text_lower = text.lower()
        
        # Define toxic keywords and phrases
        toxic_patterns = {
            'insult': [
                'stupid', 'idiot', 'moron', 'dumb', 'fool', 'asshole', 'bastard',
                'bitch', 'whore', 'slut', 'retard', 'cripple', 'fat', 'ugly'
            ],
            'threat': [
                'kill you', 'hurt you', 'beat you', 'attack', 'destroy', 'eliminate',
                'you\'ll pay', 'you\'ll regret', 'consequences', 'punish'
            ],
            'obscene': [
                'fuck', 'shit', 'piss', 'cunt', 'cock', 'dick', 'pussy', 'ass',
                'bitch', 'whore', 'slut', 'bastard'
            ],
            'hate': [
                'hate', 'despise', 'loathe', 'abhor', 'disgusting', 'repulsive',
                'worthless', 'useless', 'pathetic', 'disgusting'
            ]
        }
        
        category_scores = {}
        total_score = 0.0
        
        for category, keywords in toxic_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            category_score = min(matches * 0.2, 1.0)  # Cap at 1.0
            category_scores[category] = category_score
            total_score += category_score
        
        # Normalize total score
        overall_score = min(total_score / len(toxic_patterns), 1.0)
        
        return ToxicityResult(
            value=overall_score,
            categories=category_scores,
            confidence=0.5  # Lower confidence for fallback method
        )
    
    def get_toxicity_breakdown(self, text: str) -> Dict[str, float]:
        """Get detailed toxicity breakdown."""
        result = self.detect(text)
        return result.categories
    
    def is_toxic(self, text: str, threshold: float = 0.5) -> bool:
        """Check if text is toxic above threshold."""
        result = self.detect(text)
        return result.value > threshold 