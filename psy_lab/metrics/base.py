"""Base classes for metrics."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    value: float
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {} 