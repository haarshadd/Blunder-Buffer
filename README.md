# Blunder Buffer

**A self-improving, auditable ML framework for chess prediction using experience replay and Champion/Challenger model governance.**

## Overview

Blunder Buffer is an end-to-end machine learning system for chess outcome prediction. It combines historical chess data, feature engineering, multiple expert models, psychological/contextual features, a prediction ledger, experience replay, and controlled challenger-model promotion.

The project is designed around an important ML lifecycle problem: **how can a predictive system learn from its mistakes without blindly replacing a reliable model?**

Blunder Buffer addresses this with two complementary mechanisms:

- **Experience Replay** — deliberately reuses important past predictions, especially confident mistakes, recent examples, and historical anchor examples.
- **Champion/Challenger Governance** — evaluates a newly trained model against the current champion on a fresh holdout set and promotes it only when it provides a meaningful improvement without unacceptable regression.

---

## System Architecture

```text
                         BLUNDER BUFFER
                              │
              ┌───────────────┴───────────────┐
              │                               │
        DATA PIPELINE                    MODEL SYSTEM
              │                               │
              ▼                               ▼
          TWIC / PGN                    Expert Models
              │                               │
              ▼                               ▼
       SQLite Database                 OOF Predictions
              │                               │
              ▼                               ▼
      Feature Engineering             Meta Learner
              │                               │
              ▼                               ▼
    Psychological Context          Prediction Ledger
              │                               │
              └───────────────┬───────────────┘
                              ▼
                     Experience Replay
                              │
                              ▼
                     Challenger Model
                              │
                              ▼
                  Champion / Challenger
                       Evaluation
                       │         │
                     FAIL       PASS
                       │         │
                    Reject    Promote
                                  │
                                  ▼
                            New Champion
```

---

## Key Components

### 1. Data Ingestion

`workers/ingest.py` downloads TWIC PGN archives, filters games involving selected target players, and stores the relevant games and metadata in SQLite.

The ingestion pipeline is designed to process games individually rather than loading the entire archive into memory.

### 2. Feature Engineering

`workers/feature_extractor.py` constructs baseline features from historical games while avoiding future-information leakage.

Features include:

- Elo difference and Elo availability
- Head-to-head historical results
- Rest
- Rolling fatigue
- Consecutive-loss / tilt context
- Black's recent draw rate
- Historical game metadata

### 3. Chess Analysis

`workers/chess_analysis.py` uses Stockfish to calculate move-level centipawn loss and derives:

- White ACPL
- Black ACPL
- Target-game ACPL information

The analysis pipeline supports batched processing and progress persistence.

### 4. Expert Models

The system maintains multiple prediction experts whose out-of-fold predictions can be consumed by the meta learner.

The current pipeline includes expert prediction layers associated with:

- Baseline
- ECO / opening context
- Momentum

Their OOF probability outputs become part of the meta-model's deep input.

### 5. Psychological / Context Layer

The meta learner incorporates contextual differentials such as:

- Rest difference
- Fatigue difference
- Tilt difference
- Black draw-rate context
- Elo difference

These form the **wide feature path**, while expert-model probabilities form the **deep feature path**.

### 6. Prediction Ledger

`core/ledger.py` provides an append-only source of truth for model predictions.

Each prediction can later be resolved against the actual outcome. The system computes a **surprise score** based on the probability assigned to the outcome that actually occurred.

This makes model mistakes measurable and reusable rather than simply discarded.

### 7. Experience Replay

`core/replay_buffer.py` builds training batches from three complementary slices:

```text
40%  Prioritized experiences
     → high-surprise / confident mistakes

40%  Recent random experiences
     → current distribution

20%  Historical anchors
     → older examples for stability
```

The resulting batch is deduplicated and shuffled before training.

This design aims to reduce catastrophic forgetting while still allowing the system to learn from recent and important failures.

### 8. Wide & Deep Challenger

`train_meta.py` trains a PyTorch Wide & Deep neural network.

The architecture contains:

```text
Expert OOF Probabilities
          │
          ▼
     Deep Path
   Linear → ReLU
       → Dropout
          │
          ├──────────────┐
          │              │
          │       Wide Context Features
          │              │
          └──────┬───────┘
                 ▼
             Combined
                 │
          Linear → ReLU
              → Dropout
                 │
                 ▼
        3-Class Prediction
```

The output represents the three chess result classes used by the training pipeline.

The challenger is evaluated using **log loss** on a separate fresh holdout window before promotion.

---

## Champion / Challenger Governance

The model lifecycle is controlled by `core/orchestrator.py` and `core/registry.py`.

A challenger is not promoted merely because it trains successfully.

The evaluation flow is:

```text
Train Challenger
      │
      ▼
Fresh Holdout Evaluation
      │
      ▼
Compare Log Loss
      │
      ├── Insufficient improvement ──► Reject
      │
      ├── Regression on protected anchors ──► Reject
      │
      └── Meaningful improvement
                    │
                    ▼
                 Promote
                    │
                    ▼
             Model Registry
```

The test pipeline explicitly verifies the minimum-improvement margin used to prevent promotion based on tiny changes that may be statistical noise.

---

## Model Registry

`core/registry.py` maintains model lineage and promotion state.

The registry records information such as:

- Model version
- Sport
- Model layer
- Training window
- Evaluation metrics
- Registration time
- Promotion time
- Champion status

This provides an auditable history of model evolution.

---

## Anomaly Investigation

`investigate_anomaly.py` provides a targeted diagnostic for extreme surprise-score predictions.

It queries the prediction ledger together with baseline features to investigate cases where the model assigned an extremely small probability to the actual outcome.

This is useful for distinguishing:

- genuinely surprising outcomes
- model blind spots
- probability-calibration problems
- numerical/probability-floor effects

---

## Testing

`test_pipeline.py` exercises important system-level behavior rather than only testing isolated functions.

It checks:

1. Scoped Experience Replay sampling
2. Surprise-score retrieval
3. Champion/Challenger evaluation
4. Minimum-improvement / noise-margin behavior

Example configuration uses a 0.002 minimum improvement threshold, allowing the system to reject a challenger whose apparent gain is too small.

---

## Project Structure

```text
Blunder-Buffer/
│
├── core/
│   ├── backfill_ledger.py
│   ├── ledger.py
│   ├── orchestrator.py
│   ├── registry.py
│   └── replay_buffer.py
│
├── workers/
│   ├── ingest.py
│   ├── feature_extractor.py
│   └── chess_analysis.py
│
├── notebooks/
│   └── models.ipynb
│
├── others/
│   └── twic_historical_loop.py
│
├── investigate_anomaly.py
├── test_pipeline.py
├── train_meta.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Data and Model Artifacts

Large and generated artifacts are intentionally excluded from version control.

Examples include:

- SQLite databases
- Raw PGN/CSV datasets
- Trained model artifacts
- Stockfish executables
- Generated logs and documents
- Python cache files
- Virtual environments

This keeps the repository lightweight while preserving the source code required to understand and reproduce the pipeline.

---

## Requirements

The project uses Python with libraries including:

- pandas
- NumPy
- PyTorch
- XGBoost
- scikit-learn
- SciPy
- Matplotlib
- python-chess
- requests

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

The repository is organized around separate stages rather than a single monolithic script.

Typical workflow:

```text
1. Ingest historical games
        ↓
2. Build baseline/context features
        ↓
3. Run chess-engine analysis
        ↓
4. Generate expert predictions
        ↓
5. Maintain / backfill prediction ledger
        ↓
6. Build Experience Replay batch
        ↓
7. Train Wide & Deep challenger
        ↓
8. Evaluate on fresh holdout
        ↓
9. Champion / Challenger decision
        ↓
10. Registry update
```

Before running the full pipeline, make sure the required local data and Stockfish executable are available in the expected locations. These artifacts are deliberately not committed to the repository.

---

## Design Principles

### Learn from mistakes, not just new data

High-surprise predictions contain valuable information. The replay system ensures important failures can influence subsequent training.

### Preserve historical stability

A system that only trains on recent data can forget older behavior. Historical anchor sampling helps maintain long-term calibration.

### Never promote blindly

A challenger must demonstrate meaningful improvement and pass regression checks before becoming the new champion.

### Keep an audit trail

Predictions, outcomes, surprise scores, and model versions are recorded so model behavior can be inspected after the fact.

### Separate experimentation from governance

Training a model and deciding whether that model should become production champion are treated as different responsibilities.

---

## Current Scope

Blunder Buffer is currently focused on **chess prediction**, while several core components are intentionally designed around reusable model-governance concepts such as sport/model scoping, prediction ledgers, replay sampling, registries, and challenger evaluation.

---

## Project Status

**Research / experimental production pipeline**

The system contains an implemented ingestion pipeline, feature engineering, chess-engine analysis, prediction ledger, experience replay, neural meta-learning, anomaly investigation, and Champion/Challenger governance.

Future work can include live inference, automated deployment, richer monitoring, calibration dashboards, and broader validation across additional datasets.

---

## License

No license has been specified yet.
