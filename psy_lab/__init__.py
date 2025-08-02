"""Social Psych Arena - Large-Scale In-Silico Social-Psychology Experiments with LLM Ensembles."""

__version__ = "0.1.0"
__author__ = "Social Psych Arena Team"

from .dsl_parser import Scenario, Role, Metric
from .orchestrator import ExperimentOrchestrator
from .metrics.metric_engine import MetricEngine

__all__ = ["Scenario", "Role", "Metric", "ExperimentOrchestrator", "MetricEngine"] 