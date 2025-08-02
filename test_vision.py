#!/usr/bin/env python3
"""
Test script for the vision-enabled Asch experiment.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psy_lab.dsl_parser import load_scenario
from psy_lab.orchestrator import ExperimentOrchestrator

async def test_vision_experiment():
    """Test the vision-enabled Asch experiment."""
    
    # Load the vision scenario
    scenario = load_scenario("psy_lab/scenarios/asch_vision.yaml")
    
    print(f"Scenario: {scenario.name}")
    print(f"Description: {scenario.description}")
    print(f"Vision model required: {scenario.config.get('vision_model_required', False)}")
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(scenario, "test_logs")
    
    # Check if it's a vision experiment
    if orchestrator._is_vision_experiment():
        print("✓ This is a vision experiment")
        
        # Check if image generation is available
        if hasattr(orchestrator, 'trial_images'):
            print(f"✓ Generated {len(orchestrator.trial_images)} trial images")
            
            # Show first trial
            if orchestrator.trial_images:
                image_data, correct_answer = orchestrator.trial_images[0]
                print(f"✓ First trial correct answer: {correct_answer}")
                
                # Save test image
                import base64
                with open("test_vision_image.png", "wb") as f:
                    f.write(base64.b64decode(image_data))
                print("✓ Saved test image as test_vision_image.png")
        else:
            print("✗ No trial images generated")
    else:
        print("✗ This is not a vision experiment")
    
    # Check agents
    print(f"\nAgents:")
    for agent_id, agent in orchestrator.agents.items():
        is_vision = agent._is_vision_model()
        print(f"  {agent_id}: {agent.role.model} {'(vision)' if is_vision else '(text-only)'}")

if __name__ == "__main__":
    asyncio.run(test_vision_experiment()) 