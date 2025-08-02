"""DSL Parser for Social Psych Arena scenarios."""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import BaseModel, Field, validator


class MetricType(str, Enum):
    """Supported metric types."""
    TOXICITY = "toxicity"
    SENTIMENT = "sentiment"
    CONFORMITY = "conformity"
    COMPLIANCE = "compliance"
    OBEDIENCE = "obedience"


class StopCriteria(BaseModel):
    """Stop criteria for experiments."""
    max_turns: int = Field(default=30, description="Maximum number of turns")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens per run")
    max_cost: Optional[float] = Field(default=None, description="Maximum cost in USD")
    toxicity_threshold: Optional[float] = Field(default=None, description="Stop if toxicity exceeds threshold")


class Role(BaseModel):
    """Role definition for agents."""
    name: str = Field(..., description="Role name")
    system_prompt: str = Field(..., description="System prompt for this role")
    count: int = Field(default=1, description="Number of agents with this role")
    model: Optional[str] = Field(default=None, description="Specific model for this role")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens per response")
    
    @validator('count')
    def validate_count(cls, v):
        if v < 1:
            raise ValueError("Role count must be at least 1")
        return v


class Metric(BaseModel):
    """Metric configuration."""
    type: MetricType = Field(..., description="Metric type")
    threshold: Optional[float] = Field(default=None, description="Threshold for this metric")
    enabled: bool = Field(default=True, description="Whether this metric is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class Hierarchy(BaseModel):
    """Role hierarchy definition."""
    relationships: List[str] = Field(..., description="List of 'role1 > role2' relationships")
    
    @validator('relationships')
    def validate_relationships(cls, v):
        for rel in v:
            if '>' not in rel:
                raise ValueError(f"Invalid hierarchy relationship: {rel}")
        return v
    
    def get_hierarchy_dict(self) -> Dict[str, List[str]]:
        """Convert relationships to hierarchy dictionary."""
        hierarchy = {}
        for rel in self.relationships:
            superior, inferior = rel.split('>')
            superior = superior.strip()
            inferior = inferior.strip()
            if superior not in hierarchy:
                hierarchy[superior] = []
            hierarchy[superior].append(inferior)
        return hierarchy


class Scenario(BaseModel):
    """Complete experiment scenario."""
    name: str = Field(..., description="Scenario name")
    description: str = Field(..., description="Scenario description")
    roles: List[Role] = Field(..., description="List of roles")
    hierarchy: Optional[Hierarchy] = Field(default=None, description="Role hierarchy")
    stop_criteria: StopCriteria = Field(default_factory=StopCriteria, description="Stop criteria")
    metrics: List[Metric] = Field(default_factory=list, description="Metrics to track")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")
    
    @validator('roles')
    def validate_roles(cls, v):
        if not v:
            raise ValueError("At least one role must be defined")
        role_names = [role.name for role in v]
        if len(role_names) != len(set(role_names)):
            raise ValueError("Role names must be unique")
        return v
    
    def get_total_agents(self) -> int:
        """Get total number of agents in this scenario."""
        return sum(role.count for role in self.roles)
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """Get role by name."""
        for role in self.roles:
            if role.name == name:
                return role
        return None


def load_scenario(yaml_path: str) -> Scenario:
    """Load scenario from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return Scenario(**data)


def save_scenario(scenario: Scenario, yaml_path: str) -> None:
    """Save scenario to YAML file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(scenario.dict(), f, default_flow_style=False, indent=2) 