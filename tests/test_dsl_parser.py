"""Tests for DSL parser."""

import pytest
import tempfile
import yaml
from pathlib import Path

from psy_lab.dsl_parser import (
    Scenario, Role, Metric, MetricType, StopCriteria, 
    Hierarchy, load_scenario, save_scenario
)


def test_role_creation():
    """Test role creation and validation."""
    role = Role(
        name="test_role",
        system_prompt="You are a test role",
        count=2,
        temperature=0.7
    )
    
    assert role.name == "test_role"
    assert role.count == 2
    assert role.temperature == 0.7


def test_role_validation():
    """Test role validation."""
    with pytest.raises(ValueError):
        Role(
            name="test_role",
            system_prompt="You are a test role",
            count=0  # Invalid count
        )


def test_metric_creation():
    """Test metric creation."""
    metric = Metric(
        type=MetricType.TOXICITY,
        threshold=0.8,
        enabled=True
    )
    
    assert metric.type == MetricType.TOXICITY
    assert metric.threshold == 0.8
    assert metric.enabled is True


def test_hierarchy_creation():
    """Test hierarchy creation."""
    hierarchy = Hierarchy(relationships=["guard > prisoner", "warden > guard"])
    
    assert len(hierarchy.relationships) == 2
    hierarchy_dict = hierarchy.get_hierarchy_dict()
    assert "guard" in hierarchy_dict
    assert "prisoner" in hierarchy_dict["guard"]


def test_hierarchy_validation():
    """Test hierarchy validation."""
    with pytest.raises(ValueError):
        Hierarchy(relationships=["invalid_relationship"])


def test_scenario_creation():
    """Test scenario creation."""
    scenario = Scenario(
        name="Test Scenario",
        description="A test scenario",
        roles=[
            Role(name="role1", system_prompt="You are role1", count=1),
            Role(name="role2", system_prompt="You are role2", count=2)
        ]
    )
    
    assert scenario.name == "Test Scenario"
    assert len(scenario.roles) == 2
    assert scenario.get_total_agents() == 3


def test_scenario_validation():
    """Test scenario validation."""
    with pytest.raises(ValueError):
        Scenario(
            name="Test Scenario",
            description="A test scenario",
            roles=[
                Role(name="role1", system_prompt="You are role1", count=1),
                Role(name="role1", system_prompt="You are role1 again", count=1)  # Duplicate name
            ]
        )


def test_load_save_scenario():
    """Test loading and saving scenarios."""
    scenario = Scenario(
        name="Test Scenario",
        description="A test scenario",
        roles=[
            Role(name="role1", system_prompt="You are role1", count=1)
        ],
        seed=42
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save scenario
        save_scenario(scenario, temp_path)
        
        # Load scenario
        loaded_scenario = load_scenario(temp_path)
        
        assert loaded_scenario.name == scenario.name
        assert loaded_scenario.seed == scenario.seed
        assert len(loaded_scenario.roles) == len(scenario.roles)
        
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_stop_criteria():
    """Test stop criteria."""
    stop_criteria = StopCriteria(
        max_turns=30,
        max_tokens=1000,
        max_cost=0.50
    )
    
    assert stop_criteria.max_turns == 30
    assert stop_criteria.max_tokens == 1000
    assert stop_criteria.max_cost == 0.50 