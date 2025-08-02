# Project Requirement Document

## Project: **Social Psych Arena**

### Subtitle: *Large‑Scale In‑Silico Social‑Psychology Experiments with LLM Ensembles*

---

### 1. Background

Classic social‑psychology studies (Stanford Prison, Milgram, Asch) revealed profound effects of **power asymmetry** and **peer pressure** on human behaviour.  LLMs are increasingly embedded in multi‑agent workflows—yet their emergent social dynamics remain under‑explored.\
Social Psych Arena offers a controlled framework to replicate these experiments with ensembles of LLM “personas,” measuring linguistic cruelty, obedience, and conformity.

### 2. Vision & Value Proposition

- Provide the first **open metric suite** for social pathologies in LLM collectives.
- Generate quantitative evidence informing scalable oversight: if models show undesirable convergence under authority, alignment strategies must address inter‑agent interactions, not just single‑agent honesty.

### 3. Key Experiments & Metrics

| Experiment            | Setup                                                    | Primary Metrics                                                         |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Stanford Prison**   | 3 Guard agents, 3 Prisoner agents, warden GPT moderator  | *Cruelty Index* (toxic phrasing count / turn); Prisoner compliance rate |
| **Milgram Obedience** | Commander → Operator chain triggering `shock(level)` API | Maximum shock level; refusal point                                      |
| **Asch Conformity**   | 5 Confederates + 1 Subject answering simple perceptual Q | Switch‑to‑majority probability vs. #confederates                        |

### 4. Functional Requirements

1. **Scenario DSL**
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
2. **Agent Orchestrator**
   - Spawns n LLM subprocesses (HF or OpenAI).
   - Maintains role context + hierarchy rules.
   - Supports function calls (e.g., `shock(level)` logged as side‑effect).
3. **Metric Engine**
   - Toxicity (Perspective API or Open‑sourced detoxifier).
   - Sentiment; power‑assertion regex; compliance detectors.
   - Pluggable user‑defined Python metrics.
4. **Result Store**
   - Parquet + SQLite for runs; metadata (model, seed, cost).
5. **Visualization Dashboard**
   - Streamlit: timeline of toxicity scores; violin plots of shock level distributions; conformity curves.
6. **Cost Control**
   - Token budget per run; abort if exceeded.

### 5. Non‑Functional Requirements

- **Reproducibility** – `seed` parameter yields identical conversation trees.
- **Scalability** – Parallel execution on local CPU clusters (`--workers N`).
- **Ethics** – Sensitive content redacted when exporting logs.

### 6. System Architecture

```
psy-lab/
├── orchestrator.py     # core runtime
├── dsl_parser.py       # YAML → Scenario object
├── metrics/
│   ├── toxicity.py
│   ├── sentiment.py
│   └── conformity.py
├── dashboards/
│   └── app.py          # Streamlit UI
├── scenarios/
│   ├── prison.yaml
│   ├── milgram.yaml
│   └── asch.yaml
└── tests/
```

**Runtime Flow** `yaml` → Scenario → Orchestrator → Agents (LLMs) ↔ Metric hooks → Parquet logs → Dashboard

### 7. Deployment & Usage

```bash
pip install psy-lab
psy-lab run scenarios/prison.yaml --model mistral-7b --workers 6
psy-lab dashboard logs/*.parquet
```

### 8. Milestone Plan (8 Weeks)

| Week | Deliverable                            |
| ---- | -------------------------------------- |
| 1    | DSL schema + Asch minimal case         |
| 2    | Orchestrator core (local HF models)    |
| 3    | Toxicity + sentiment metric modules    |
| 4    | Milgram experiment with function‑calls |
| 5    | Stanford Prison multi‑turn experiment  |
| 6    | Streamlit dashboard MVP                |
| 7    | Parallel run harness + cost caps       |
| 8    | Docs, tutorial, v0.1 tag               |

### 9. Risks & Mitigation

| Risk                              | Impact     | Mitigation                                    |
| --------------------------------- | ---------- | --------------------------------------------- |
| API cost blow‑up (if OpenAI)      | Budget     | Local GGUF or llama.cpp models default        |
| Toxic logs unethical to share     | Compliance | Redaction pipeline; share metrics only        |
| Scenario complexity overwhelms UI | Confusion  | Start with 3 flagship experiments; modularise |

### 10. Acceptance Criteria

- Reproduce Asch baseline: ≥ 70 % conformity with 5 confederates using GPT‑4 baseline.
- Dashboard shows cruelty index rising over turns in Prison scenario when using less‑aligned model.
- `psy-lab run ...` completes under cost cap and writes Parquet + summary JSON.

