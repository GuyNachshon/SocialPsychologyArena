# Social Psych Arena

Large-Scale In-Silico Social-Psychology Experiments with LLM Ensembles

## Overview

Social Psych Arena replicates classic social psychology experiments (Stanford Prison, Milgram, Asch) using ensembles of LLM "personas" to study emergent social dynamics in multi-agent systems.

## Key Features

- **Controlled Experiments**: Replicate classic social psychology studies with LLM agents
- **Quantitative Metrics**: Measure toxicity, obedience, conformity, and power dynamics
- **Reproducible Results**: Seed-based execution for consistent outcomes
- **Cost Control**: Configurable token budgets and parallel execution
- **Interactive Dashboard**: Visualize experiment results and trends

## Quick Start

```bash
# Install
pip install psy-lab

# Run Stanford Prison experiment
psy-lab run scenarios/prison.yaml --model Qwen/Qwen3-0.6B

# View results dashboard
psy-lab dashboard logs/*.parquet
```

## Experiments

### Stanford Prison Experiment
- **Setup**: 3 Guard agents, 3 Prisoner agents, 1 Warden moderator
- **Metrics**: Cruelty Index, Prisoner compliance rate
- **Goal**: Study power asymmetry effects on behavior

### Milgram Obedience Experiment
- **Setup**: Commander → Operator chain with shock function calls
- **Metrics**: Maximum shock level, refusal point
- **Goal**: Measure obedience under authority

### Asch Conformity Experiment
- **Setup**: 5 Confederates + 1 Subject answering perceptual questions
- **Metrics**: Switch-to-majority probability
- **Goal**: Study peer pressure and conformity

## Architecture

```
psy-lab/
├── orchestrator.py     # LangGraph-based runtime
├── dsl_parser.py       # YAML scenario parser
├── metrics/            # Analysis modules
├── dashboards/         # Streamlit visualization
├── scenarios/          # Experiment definitions
└── tests/             # Test suite
```

## Configuration

Experiments are defined using YAML:

```yaml
roles:
  - name: guard
    system_prompt: "You are a prison guard..."
    count: 3
hierarchy:
  guard > prisoner
stop_criteria:
  max_turns: 30
metrics:
  - type: toxicity
    threshold: 0.8
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black psy_lab/
isort psy_lab/
```

## License

MIT License 