# Social Psych Arena - Quick Start Guide

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Social-psychAgentArena
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Quick Test

Run a simple test experiment to verify everything works:

```bash
python example.py
```

This will run a 2-agent conversation with sentiment and toxicity tracking.

## Running Experiments

### 1. List Available Scenarios

```bash
psy-lab list-scenarios
```

### 2. Validate a Scenario

```bash
psy-lab validate psy_lab/scenarios/asch.yaml
```

### 3. Run an Experiment

```bash
# Run Asch conformity experiment
psy-lab run psy_lab/scenarios/asch.yaml

# Run with custom model
psy-lab run psy_lab/scenarios/milgram.yaml --model Qwen/Qwen3-0.6B

# Run with custom parameters
psy-lab run psy_lab/scenarios/prison.yaml --max-turns 20 --seed 123
```

### 4. View Results Dashboard

```bash
# Launch dashboard for the most recent results
psy-lab dashboard

# Or specify a specific results directory
psy-lab dashboard logs/Stanford_Prison_Experiment_20241201_143022/
```

## Available Experiments

### Asch Conformity Experiment
- **File:** `psy_lab/scenarios/asch.yaml`
- **Setup:** 5 confederates + 1 subject
- **Goal:** Measure conformity under peer pressure
- **Metrics:** Conformity rate, sentiment analysis

### Milgram Obedience Experiment
- **File:** `psy_lab/scenarios/milgram.yaml`
- **Setup:** Commander → Operator → Learner chain
- **Goal:** Measure obedience to authority
- **Metrics:** Obedience rate, maximum shock level, toxicity

### Stanford Prison Experiment
- **File:** `psy_lab/scenarios/prison.yaml`
- **Setup:** 3 guards + 3 prisoners + 1 warden
- **Goal:** Study power dynamics and dehumanization
- **Metrics:** Cruelty index, compliance rate, toxicity

## Configuration

### Model Selection
- **Local models:** Use HuggingFace model names (e.g., `Qwen/Qwen3-0.6B`)
- **OpenAI models:** Use `gpt-4` or `gpt-3.5-turbo` (requires API key)

### Cost Control
- Set `max_cost` in scenarios or use `--max-cost` CLI option
- Monitor token usage in logs

### Reproducibility
- Use `--seed` parameter for reproducible results
- Seeds are automatically saved in metadata

## Understanding Results

### Output Files
- `conversation.parquet`: Full conversation history
- `metrics.parquet`: Calculated metrics over time
- `metadata.json`: Experiment configuration and summary

### Key Metrics
- **Toxicity:** Measures harmful content (0-1 scale)
- **Sentiment:** Overall emotional tone (-1 to 1 scale)
- **Conformity:** Rate of agreement with majority
- **Compliance:** Rate of following orders
- **Obedience:** Maximum authority level reached

### Dashboard Features
- **Overview:** Experiment metadata and summary statistics
- **Conversation:** Timeline visualization and message excerpts
- **Metrics:** Time series plots and distributions
- **Analysis:** Correlation analysis and advanced insights

## Troubleshooting

### Common Issues

1. **Model loading fails:**
   - Ensure you have enough GPU memory for local models
   - Try smaller models like `Qwen/Qwen3-0.6B`
   - Use CPU-only models if GPU is unavailable

2. **Memory issues:**
   - Reduce `max_turns` in scenarios
   - Use smaller models
   - Increase system swap space

3. **Slow performance:**
   - Use smaller models for testing
   - Reduce number of agents
   - Use GPU acceleration if available

### Getting Help

- Check the logs in `psy_lab.log`
- Run with `--verbose` flag for detailed output
- Validate scenarios before running

## Next Steps

1. **Create custom scenarios:** Modify existing YAML files or create new ones
2. **Add custom metrics:** Extend the metrics engine with new analysis
3. **Scale experiments:** Run multiple experiments in parallel
4. **Analyze results:** Use the dashboard for detailed analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details. 