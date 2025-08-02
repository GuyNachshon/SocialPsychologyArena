#!/usr/bin/env python3
"""Simple example of running a Social Psych Arena experiment."""

import asyncio
import logging
from pathlib import Path

from psy_lab.dsl_parser import load_scenario
from psy_lab.orchestrator import ExperimentOrchestrator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_simple_experiment():
    """Run a simple experiment to test the system."""
    
    # Create a simple scenario programmatically
    from psy_lab.dsl_parser import Scenario, Role, Metric, MetricType, StopCriteria
    
    scenario = Scenario(
        name="Simple Test Experiment",
        description="A simple test to verify the system works",
        roles=[
            Role(
                name="speaker1",
                system_prompt="You are a friendly person. Keep responses short and positive.",
                count=1,
                temperature=0.7,
                max_tokens=50
            ),
            Role(
                name="speaker2", 
                system_prompt="You are a neutral person. Respond to what others say.",
                count=1,
                temperature=0.7,
                max_tokens=50
            )
        ],
        stop_criteria=StopCriteria(max_turns=5, max_tokens=500),
        metrics=[
            Metric(type=MetricType.SENTIMENT, enabled=True),
            Metric(type=MetricType.TOXICITY, enabled=True)
        ],
        seed=42
    )
    
    logger.info(f"Running experiment: {scenario.name}")
    logger.info(f"Total agents: {scenario.get_total_agents()}")
    
    # Create orchestrator
    orchestrator = ExperimentOrchestrator(scenario, output_dir="test_logs")
    
    try:
        # Run experiment
        results = await orchestrator.run()
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {results['results_dir']}")
        logger.info(f"Total turns: {results['final_state']['turn']}")
        logger.info(f"Stop reason: {results['final_state']['stop_reason']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def main():
    """Main function."""
    print("🧪 Social Psych Arena - Simple Test")
    print("=" * 40)
    
    try:
        # Run the experiment
        results = asyncio.run(run_simple_experiment())
        
        print("\n✅ Experiment completed successfully!")
        print(f"📁 Results saved to: {results['results_dir']}")
        
        # Show some basic results
        final_state = results['final_state']
        print(f"🔄 Total turns: {final_state['turn']}")
        print(f"🛑 Stop reason: {final_state['stop_reason']}")
        
        if final_state['metrics']:
            print("\n📊 Metrics summary:")
            for metric_name, values in final_state['metrics'].items():
                if values:
                    avg = sum(values) / len(values)
                    print(f"  • {metric_name}: {avg:.3f} (avg)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 