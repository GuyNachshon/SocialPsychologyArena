"""Sentiment analysis for social psychology experiments."""

import logging
import re
from typing import Dict, Any, List
from dataclasses import dataclass

from .base import MetricResult

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult(MetricResult):
    """Result of sentiment analysis."""
    polarity: float = 0.0  # -1 to 1
    subjectivity: float = 0.0  # 0 to 1
    emotions: Dict[str, float] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.emotions is None:
            self.emotions = {}


class SentimentAnalyzer:
    """Analyze sentiment in text using lexicon-based approach."""
    
    def __init__(self):
        self._load_lexicons()
    
    def _load_lexicons(self):
        """Load sentiment lexicons."""
        # Positive and negative word lists
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome',
            'love', 'like', 'enjoy', 'happy', 'joy', 'pleasure', 'satisfaction',
            'success', 'win', 'victory', 'achievement', 'accomplishment',
            'beautiful', 'gorgeous', 'stunning', 'perfect', 'ideal', 'best',
            'kind', 'gentle', 'caring', 'compassionate', 'understanding',
            'peaceful', 'calm', 'relaxed', 'content', 'grateful', 'thankful'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'disgusting',
            'hate', 'dislike', 'loathe', 'despise', 'angry', 'furious', 'rage',
            'sad', 'depressed', 'miserable', 'unhappy', 'disappointed',
            'failure', 'lose', 'defeat', 'loss', 'mistake', 'error',
            'ugly', 'hideous', 'repulsive', 'disgusting', 'terrible',
            'cruel', 'mean', 'harsh', 'brutal', 'violent', 'aggressive',
            'anxious', 'worried', 'scared', 'fearful', 'terrified', 'panic'
        }
        
        # Emotion categories
        self.emotion_words = {
            'anger': {'angry', 'furious', 'rage', 'mad', 'irritated', 'annoyed', 'frustrated'},
            'fear': {'scared', 'afraid', 'terrified', 'fearful', 'anxious', 'worried', 'panic'},
            'sadness': {'sad', 'depressed', 'miserable', 'unhappy', 'grief', 'sorrow', 'melancholy'},
            'joy': {'happy', 'joyful', 'excited', 'thrilled', 'elated', 'ecstatic', 'delighted'},
            'surprise': {'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered'},
            'disgust': {'disgusted', 'repulsed', 'revolted', 'sickened', 'appalled', 'horrified'},
            'trust': {'trust', 'confident', 'secure', 'safe', 'reliable', 'dependable', 'faithful'},
            'anticipation': {'excited', 'eager', 'hopeful', 'optimistic', 'enthusiastic', 'keen'}
        }
        
        # Intensifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'incredibly': 2.0,
            'absolutely': 2.0, 'completely': 1.8, 'totally': 1.8, 'utterly': 2.0,
            'slightly': 0.5, 'somewhat': 0.7, 'kind of': 0.6, 'sort of': 0.6,
            'not': -1.0, 'never': -1.5, 'no': -1.0, 'none': -1.0
        }
    
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment in text."""
        if not text or not text.strip():
            return SentimentResult(value=0.0, polarity=0.0, subjectivity=0.0)
        
        try:
            # Tokenize text
            words = self._tokenize(text.lower())
            
            # Calculate polarity
            polarity = self._calculate_polarity(words)
            
            # Calculate subjectivity
            subjectivity = self._calculate_subjectivity(words)
            
            # Detect emotions
            emotions = self._detect_emotions(words)
            
            # Overall sentiment score (normalized polarity)
            sentiment_score = (polarity + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            return SentimentResult(
                value=sentiment_score,
                polarity=polarity,
                subjectivity=subjectivity,
                emotions=emotions,
                confidence=0.7
            )
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return SentimentResult(value=0.5, polarity=0.0, subjectivity=0.0)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _calculate_polarity(self, words: List[str]) -> float:
        """Calculate polarity score (-1 to 1)."""
        score = 0.0
        total_words = len(words)
        
        if total_words == 0:
            return 0.0
        
        for i, word in enumerate(words):
            word_score = 0.0
            
            # Check positive words
            if word in self.positive_words:
                word_score = 1.0
            # Check negative words
            elif word in self.negative_words:
                word_score = -1.0
            
            # Apply intensifiers
            if i > 0 and words[i-1] in self.intensifiers:
                multiplier = self.intensifiers[words[i-1]]
                word_score *= multiplier
            
            score += word_score
        
        # Normalize by number of words
        return max(-1.0, min(1.0, score / total_words))
    
    def _calculate_subjectivity(self, words: List[str]) -> float:
        """Calculate subjectivity score (0 to 1)."""
        subjective_words = self.positive_words.union(self.negative_words)
        subjective_count = sum(1 for word in words if word in subjective_words)
        
        return subjective_count / len(words) if words else 0.0
    
    def _detect_emotions(self, words: List[str]) -> Dict[str, float]:
        """Detect emotions in text."""
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_words.keys()}
        
        for word in words:
            for emotion, emotion_words in self.emotion_words.items():
                if word in emotion_words:
                    emotion_scores[emotion] += 1.0
        
        # Normalize emotion scores
        total_words = len(words)
        if total_words > 0:
            emotion_scores = {
                emotion: score / total_words 
                for emotion, score in emotion_scores.items()
            }
        
        return emotion_scores
    
    def get_dominant_emotion(self, text: str) -> str:
        """Get the dominant emotion in text."""
        result = self.analyze(text)
        if not result.emotions:
            return "neutral"
        
        dominant_emotion = max(result.emotions.items(), key=lambda x: x[1])
        return dominant_emotion[0] if dominant_emotion[1] > 0 else "neutral"
    
    def is_positive(self, text: str, threshold: float = 0.6) -> bool:
        """Check if text is positive."""
        result = self.analyze(text)
        return result.value > threshold
    
    def is_negative(self, text: str, threshold: float = 0.4) -> bool:
        """Check if text is negative."""
        result = self.analyze(text)
        return result.value < threshold 