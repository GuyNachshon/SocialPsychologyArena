"""Command-line interface for Social Psych Arena."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .dsl_parser import load_scenario
from .orchestrator import ExperimentOrchestrator

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('psy_lab.log')
        ]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool):
    """Social Psych Arena - Large-Scale In-Silico Social-Psychology Experiments."""
    setup_logging(verbose)


@main.command()
@click.argument('scenario_file', type=click.Path(exists=True))
@click.option('--model', '-m', help='Model to use for all agents (overrides scenario settings)')
@click.option('--output-dir', '-o', default='logs', help='Output directory for results')
@click.option('--seed', '-s', type=int, help='Random seed for reproducibility')
@click.option('--max-turns', type=int, help='Override max turns from scenario')
@click.option('--max-cost', type=float, help='Override max cost from scenario')
def run(scenario_file: str, model: Optional[str], output_dir: str, seed: Optional[int], 
        max_turns: Optional[int], max_cost: Optional[float]):
    """Run a social psychology experiment."""
    
    console.print(f"[bold blue]Loading scenario from {scenario_file}[/bold blue]")
    
    try:
        # Load scenario
        scenario = load_scenario(scenario_file)
        
        # Override settings if provided
        if seed is not None:
            scenario.seed = seed
        if max_turns is not None:
            scenario.stop_criteria.max_turns = max_turns
        if max_cost is not None:
            scenario.stop_criteria.max_cost = max_cost
        
        # Override model for all roles if specified
        if model is not None:
            for role in scenario.roles:
                role.model = model
            console.print(f"[green]✓[/green] Using model: {model} for all agents")
        else:
            # Show models being used
            models_used = set(role.model or "default" for role in scenario.roles)
            console.print(f"[green]✓[/green] Models: {', '.join(models_used)}")
        
        console.print(f"[green]✓[/green] Loaded scenario: {scenario.name}")
        console.print(f"[green]✓[/green] Total agents: {scenario.get_total_agents()}")
        
        # Create orchestrator
        orchestrator = ExperimentOrchestrator(scenario, output_dir)
        
        # Run experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running experiment...", total=None)
            
            # Run the experiment
            results = asyncio.run(orchestrator.run())
            
            progress.update(task, description="Experiment completed!")
        
        # Display results
        display_results(results, scenario)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)


@main.command()
@click.argument('results_dir', type=click.Path(exists=True))
def dashboard(results_dir: str):
    """Launch the Streamlit dashboard for viewing results."""
    import subprocess
    import sys
    
    dashboard_path = Path(__file__).parent / "dashboards" / "app.py"
    
    if not dashboard_path.exists():
        console.print("[bold red]Dashboard not found. Please install the package first.[/bold red]")
        sys.exit(1)
    
    console.print(f"[bold blue]Launching dashboard for {results_dir}[/bold blue]")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(dashboard_path),
            "--", results_dir
        ])
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error launching dashboard: {e}[/bold red]")
        sys.exit(1)


@main.command()
@click.argument('scenario_file', type=click.Path(exists=True))
def validate(scenario_file: str):
    """Validate a scenario file."""
    try:
        scenario = load_scenario(scenario_file)
        
        table = Table(title=f"Scenario Validation: {scenario.name}")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="white")
        
        # Validate roles
        for role in scenario.roles:
            table.add_row(
                f"Role: {role.name}",
                "✓ Valid",
                f"Count: {role.count}, Model: {role.model or 'default'}"
            )
        
        # Validate hierarchy
        if scenario.hierarchy:
            for rel in scenario.hierarchy.relationships:
                table.add_row(
                    "Hierarchy",
                    "✓ Valid",
                    rel
                )
        else:
            table.add_row(
                "Hierarchy",
                "⚠ Missing",
                "No hierarchy defined"
            )
        
        # Validate metrics
        for metric in scenario.metrics:
            table.add_row(
                f"Metric: {metric.type.value}",
                "✓ Valid",
                f"Threshold: {metric.threshold or 'none'}"
            )
        
        # Validate stop criteria
        table.add_row(
            "Stop Criteria",
            "✓ Valid",
            f"Max turns: {scenario.stop_criteria.max_turns}"
        )
        
        console.print(table)
        console.print(f"[bold green]✓ Scenario is valid![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Validation failed: {e}[/bold red]")
        sys.exit(1)


@main.command()
def list_scenarios():
    """List available scenarios."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    
    if not scenarios_dir.exists():
        console.print("[bold red]No scenarios directory found[/bold red]")
        return
    
    table = Table(title="Available Scenarios")
    table.add_column("File", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white")
    
    for yaml_file in scenarios_dir.glob("*.yaml"):
        try:
            scenario = load_scenario(str(yaml_file))
            table.add_row(
                yaml_file.name,
                scenario.name,
                scenario.description[:60] + "..." if len(scenario.description) > 60 else scenario.description
            )
        except Exception as e:
            table.add_row(
                yaml_file.name,
                "[red]Error[/red]",
                str(e)
            )
    
    console.print(table)


@main.command()
def list_models():
    """List available models and presets."""
    config_file = Path("config.yaml")
    
    if not config_file.exists():
        console.print("[bold red]Config file not found[/bold red]")
        return
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Show presets
        console.print("[bold blue]Model Presets:[/bold blue]")
        presets_table = Table()
        presets_table.add_column("Preset", style="cyan")
        presets_table.add_column("Model", style="green")
        presets_table.add_column("Description", style="white")
        
        for preset_name, preset_data in config.get('models', {}).items():
            presets_table.add_row(
                preset_name,
                preset_data.get('default', 'N/A'),
                preset_data.get('description', '')
            )
        
        console.print(presets_table)
        
        # Show available models by provider
        console.print("\n[bold blue]Available Models by Provider:[/bold blue]")
        for provider, models in config.get('available_models', {}).items():
            console.print(f"\n[bold]{provider.title()}:[/bold]")
            for model in models:
                console.print(f"  • {model}")
                
    except Exception as e:
        console.print(f"[bold red]Error loading config: {e}[/bold red]")


def display_results(results: dict, scenario):
    """Display experiment results."""
    console.print("\n[bold green]Experiment Results[/bold green]")
    console.print("=" * 50)
    
    # Basic info
    console.print(f"Scenario: {scenario.name}")
    console.print(f"Results directory: {results['results_dir']}")
    
    # Final state info
    final_state = results['final_state']
    console.print(f"Total turns: {final_state['turn']}")
    console.print(f"Stop reason: {final_state['stop_reason']}")
    console.print(f"Total tokens: {final_state['total_tokens']}")
    console.print(f"Total cost: ${final_state['total_cost']:.4f}")
    
    # Metrics summary
    if final_state['metrics']:
        console.print("\n[bold]Metrics Summary:[/bold]")
        metrics_table = Table()
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Mean", style="green")
        metrics_table.add_column("Max", style="yellow")
        metrics_table.add_column("Min", style="red")
        
        for metric_name, values in final_state['metrics'].items():
            if values:
                mean_val = sum(values) / len(values)
                max_val = max(values)
                min_val = min(values)
                metrics_table.add_row(
                    metric_name,
                    f"{mean_val:.3f}",
                    f"{max_val:.3f}",
                    f"{min_val:.3f}"
                )
        
        console.print(metrics_table)
    
    # Files created
    console.print("\n[bold]Files created:[/bold]")
    console.print(f"  • Conversation: {results['conversation_file']}")
    console.print(f"  • Metrics: {results['metrics_file']}")
    console.print(f"  • Metadata: {results['metadata_file']}")


if __name__ == '__main__':
    main() 