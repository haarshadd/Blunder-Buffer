# Blunder Buffer

**A self-improving, auditable probabilistic forecasting engine for chess — built to learn from its mistakes without blindly replacing a reliable model.**

Blunder Buffer began as a historical chess prediction project and is evolving into a **real-time, tournament-aware forecasting system**.

Its first real-world deployment target is the **46th FIDE Chess Olympiad 2026**. The Olympiad is a test environment, not the architectural boundary: the core is designed around normalized chess-game events so that the same system can eventually support Swiss tournaments, round robins, Candidates/World Championship events, and other chess formats.

---

## Why Blunder Buffer?

A prediction model is useful only if its probabilities remain trustworthy over time.

Blunder Buffer therefore treats forecasting as a complete lifecycle:

```text
Historical Games
      │
      ▼
Feature Engineering
      │
      ├──────────────┬──────────────┐
      ▼              ▼              ▼
   Baseline         ECO         Momentum
    Expert         Expert         Expert
      │              │              │
      └──────────────┬──────────────┘
                     ▼
                Meta Learner
                     │
                     ▼
              Prediction Ledger
                     │
              Actual Outcome
                     │
                     ▼
              Surprise Score
                     │
                     ▼
              Experience Replay
                     │
                     ▼
             Challenger Training
                     │
                     ▼
          Champion / Challenger
               Governance
```

The system is built around two complementary ideas:

1. **Experience Replay** — important mistakes, recent examples, and historical anchors are deliberately reused during future training.
2. **Champion/Challenger Governance** — a challenger must demonstrate meaningful improvement on a fresh holdout and pass regression checks before it can replace the champion.



# Architecture

## Historical prediction pipeline

```text
TWIC / PGN
   │
   ▼
SQLite
   │
   ▼
Historical Features
   │
   ├───────────────┬────────────────┐
   ▼               ▼                ▼
Baseline          ECO            Momentum
Expert            Expert          Expert
   │               │                │
   └───────────────┼────────────────┘
                   ▼
             OOF Predictions
                   │
                   ▼
              Meta Learner
                   │
                   ▼
          Prediction Probabilities
```


---

# Live Tournament Architecture

The live system treats a tournament as a stream of **normalized game events**, rather than hard-coding a particular tournament workflow.

Different external sources can provide different parts of the event:

```text
Chess-Results
    │
    ├── Pairings
    ├── Players
    ├── Ratings
    └── Results
          │
          ▼
     ChessGameEvent
          ▲
          │
Live Broadcast / PGN
    │
    ├── Moves
    ├── ECO / opening
    └── Live game state
```


## Staged prediction lifecycle

### T0 — Pregame

When a pairing becomes known:

```text
Pairing appears
      │
      ▼
Baseline + Momentum + Context
      │
      ▼
Pregame Meta Learner
      │
      ▼
T0 Prediction
      │
      ▼
Prediction Ledger
```

T0 is produced **before the game starts** and is immutable.

### T1 — Opening-conditioned

When live game information makes an ECO/opening condition available:

```text
Live PGN / Moves
      │
      ▼
Observed ECO
      │
      ▼
ECO Expert
      │
      ├── Baseline
      ├── ECO
      └── Momentum
             │
             ▼
      Opening-aware Meta
             │
             ▼
          T1 Ledger
```

T1 is a **new prediction**, not an overwrite of T0.


### T2 — Result

When the game finishes:

```text
Final Result
     │
     ▼
Resolve Ledger Prediction
     │
     ▼
Surprise Score
     │
     ▼
Replay / Evaluation
```

The surprise score is based on the probability assigned to the outcome that actually occurred.

---

# Prediction Ledger

`core/ledger.py` provides the append-only source of truth for predictions.

Each prediction records:

- Event ID
- Sport
- Model version
- Class labels
- Predicted probabilities
- Prediction timestamp
- Actual outcome, once known
- Surprise score, once resolved

The ledger makes predictions auditable and allows mistakes to become training information instead of disappearing after inference.

The event ID is designed to be deterministic and source-independent enough for deduplication and recovery.

---

# Model System

Blunder Buffer uses multiple specialists rather than forcing every signal through a single model.

## Baseline Expert

The baseline expert captures fundamental pregame information such as:

- White Elo
- Black Elo
- Elo difference
- Elo availability
- Historical wins
- Historical draws

## ECO Expert

The ECO specialist models opening-specific information.

It uses:

- ECO code
- Player/opening familiarity
- Familiarity differential

It is an **observed-information specialist** and therefore belongs to the opening-conditioned T1 stage rather than pure pregame prediction.

## Momentum Expert

The momentum specialist models recent player-state signals including:

- Residual performance difference
- Elo trend difference
- Volatility difference
- Color-adjusted residual difference

## ACPL Specialist

The ACPL expert is a research specialist based on Stockfish-derived move-quality information.

Because move-level engine analysis is expensive and unavailable before a game begins, ACPL is not part of the initial T0 live path.

It remains a candidate specialist for future live/post-game analysis.

---

# Meta Learning

Blunder Buffer uses Wide & Deep models to combine contextual information with expert probabilities.

Conceptually:

```text
                    ┌────────────────────┐
                    │ Expert Probabilities│
                    └─────────┬──────────┘
                              ▼
                         Deep Path
                              │
Context Features ───────► Wide Path
                              │
                              ▼
                          Combined
                              │
                              ▼
                       3-Class Output
```

The output classes are:

```text
Black win
Draw
White win
```

The live system currently maintains separate meta-learning stages:

- **Pregame Wide & Deep** — T0
- **Opening-aware Wide & Deep** — T1

This prevents information that only becomes available after the game begins from leaking into the pregame forecast.

---

# Leakage Discipline

A central design requirement is that a prediction must only use information that would have been available at the prediction timestamp.

### Allowed for T0

- Historical player results
- Historical Elo
- Historical head-to-head information
- Historical rest/fatigue/context features
- Previously trained model outputs

### Not allowed for T0

- Final game result
- Future games
- Future opening information
- Moves from the game being predicted
- Post-game engine analysis

### Allowed for T1

Everything available to T0, plus information legitimately observed after the game has started, such as an observed ECO/opening condition.

This separation is fundamental to making live accuracy measurements meaningful.

---

# Experience Replay

`core/replay_buffer.py` deliberately samples from different parts of the historical experience.

```text
40%  Prioritized Experiences
     → high-surprise / important mistakes

40%  Recent Random Experiences
     → current distribution

20%  Historical Anchors
     → older examples for stability
```

The sampled training batch is deduplicated and shuffled.

Experience Replay balances adaptation to recent data with historical stability.

---

# Champion / Challenger Governance

A model is not promoted simply because it trains successfully.

`core/orchestrator.py` evaluates a challenger against the current champion.

```text
Train Challenger
       │
       ▼
Fresh Holdout
       │
       ▼
Compare Log Loss
       │
       ├── Improvement too small ──► Reject
       │
       ├── Protected-anchor regression ──► Reject
       │
       └── Meaningful improvement
                         │
                         ▼
                      Promote
                         │
                         ▼
                  Model Registry
```

The governance layer includes a **minimum improvement/noise margin** so that tiny apparent gains are not treated as meaningful improvements.

Lower log loss is better.

The registry records model lineage and promotion state so that model changes remain auditable.

---

# Model Registry

`core/registry.py` maintains model lineage.

Recorded information includes:

- Model version
- Sport
- Model layer
- Training window
- Evaluation metrics
- Registration time
- Promotion time
- Champion status


---

# Tournament Standings Forecasting

A major next capability is to move from **individual game prediction** to **tournament-level forecasting**.

This is especially important for Swiss tournaments such as the Chess Olympiad.

## MVP: Standings Prediction

The first tournament-level MVP is **not full simulation**.

The goal is:

> **Given the current tournament state and the remaining schedule/pairings, estimate the final standings distribution.**

Conceptually:

```text
Current Standings
       │
       ├── Current Scores
       ├── Board Points
       ├── Tie-break context
       └── Remaining Pairings
                  │
                  ▼
          Game-level Probabilities
                  │
                  ▼
          Expected Future Points
                  │
                  ▼
          Final Standings Forecast
```

The system should eventually answer:

- Which teams are most likely to finish first?
- Probability of finishing in the top 3 / top 10 / top N
- Expected final match points
- Expected board points
- Probability distribution over finishing positions
- How much the current round changes the standings outlook

### Full tournament simulation

Full Monte Carlo tournament simulation is a **later capability**, not the initial standings MVP.

Once the standings forecaster is reliable, simulation can repeatedly sample future game/team outcomes:

```text
Current State
     │
     ▼
Sample remaining outcomes
     │
     ▼
Recalculate standings
     │
     ▼
Repeat thousands of times
     │
     ▼
Distribution of final standings
```

The important architectural point is that **simulation should consume the forecasting engine**, not become a second independent prediction system.

---

# Streamlit Interface

The planned Streamlit UI will expose the forecasting system without requiring users to inspect SQLite databases or terminal output.

Initial dashboard concepts:

```text
┌──────────────────────────────────────────────┐
│              BLUNDER BUFFER                  │
├──────────────────────────────────────────────┤
│ Live Tournament        Round 1               │
│                                              │
│ T0 Predictions       T1 Updates              │
│ ───────────────      ───────────────         │
│ Game A               Game A                  │
│ W 50% D 26% B 24%    W 52% D 24% B 24%      │
│                                              │
├──────────────────────────────────────────────┤
│              Standings Forecast              │
│                                              │
│ Team       Current  Expected  Top-3         │
│ ...                                           │
└──────────────────────────────────────────────┘
```

The UI should eventually provide:

- Live tournament state
- Game-level predictions
- T0 → T1 prediction changes
- Completed prediction accuracy
- Surprise-score leaders
- Team standings forecast
- Model/champion information
- Historical calibration metrics
- Replay/challenger status


---

# Tournament Abstraction

Blunder Buffer should not contain an Olympiad-specific prediction engine.

The tournament layer is configuration-driven:

```python
TournamentConfig(
    tournament_id="...",
    section="open",
    source="chess_results",
    poll_seconds=60,
)
```

The event layer normalizes individual games into `ChessGameEvent` objects containing information such as:

- Tournament ID
- Section
- Round
- Board
- White / Black
- Ratings
- FIDE IDs when available
- Teams
- Scheduled time
- Status
- Result
- ECO
- Source
- Moves

This makes the live worker reusable across tournament types.

---

# Tournament Formats

The target abstraction is broader than the Olympiad.

### Swiss

```text
Round N result
      │
      ▼
Next pairing generated
      │
      ▼
Pregame prediction
      │
      ▼
Game
      │
      ▼
Result
      │
      ▼
Standings update
```

### Round Robin

Pairings may already exist as a complete schedule, but each game can still be normalized into the same event model.

### Team Tournaments

Team context becomes part of the event/standings layer while individual board games remain the basic prediction units.

### Candidates / World Championship

The same game-event and ledger concepts can be used with format-specific standings and scheduling logic.

---

# Recovery and Reliability

Live systems must survive restarts.

The worker therefore does not rely exclusively on in-memory state.

```text
Worker instance A
      │
      ▼
Prediction logged to ledger
      │
      X
   restart
      │
      ▼
Worker instance B
      │
      ▼
Same finished event observed
      │
      ▼
Existing ledger prediction found
      │
      ▼
Prediction resolved
```

This behavior is covered by a dedicated recovery test.

The ledger is the persistent source of truth; worker memory is only operational state.

---

# Data

The historical database contains a large collection of chess games and engineered features.

Large artifacts are intentionally excluded from Git.

Examples:

- SQLite databases
- Raw PGN files
- CSV datasets
- Trained model artifacts
- Stockfish executables
- Generated logs
- Documents
- Python caches
- Virtual environments


---

# Project Structure

```text
Blunder-Buffer/
│
├── core/
│   ├── backfill_ledger.py
│   ├── ledger.py
│   ├── orchestrator.py
│   ├── registry.py
│   ├── replay_buffer.py
│   └── __init__.py
│
├── live/
│   ├── config.py
│   ├── eco.py
│   ├── events.py
│   ├── features_live.py
│   ├── opening_update.py
│   ├── pregame.py
│   ├── staged_predict.py
│   ├── train_pregame_meta.py
│   └── __init__.py
│
├── workers/
│   ├── adapters/
│   │   ├── base.py
│   │   └── chess_results.py
│   ├── chess_analysis.py
│   ├── feature_extractor.py
│   ├── ingest.py
│   └── realtime_chess.py
│
├── notebooks/
│   └── models.ipynb
│
├── others/
│   └── twic_historical_loop.py
│
├── tests/
│   ├── test_chess_results_adapter.py
│   ├── test_worker_t0.py
│   └── worker_recovery.py
│
├── investigate_anomaly.py
├── test_pipeline.py
├── train_meta.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Running

Create/activate a virtual environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

For module-based worker execution, run from the repository root:

```bash
python -m workers.realtime_chess --tournament-id <TOURNAMENT_ID> --section open --round 1 --poll 60
```

The worker polls a configured source, normalizes newly observed events, creates T0 predictions for unfinished games, and resolves predictions when results become available.

---

# Current Olympiad Deployment

The **46th FIDE Chess Olympiad 2026** is the first real-world test environment for the live chess pipeline.

The deployment is intentionally treated as an adapter/configuration problem:

```text
Olympiad
   │
   ├── Chess-Results
   │      └── Pairings / results
   │
   └── Live broadcast source
          └── Moves / opening information
```


The immediate live objective is:

```text
Round 1 pairing
      ↓
T0 prediction
      ↓
Ledger
      ↓
Game result
      ↓
T0 resolution
      ↓
Surprise score
```

The next stage adds opening-conditioned T1 predictions when a reliable live move/ECO source is available.

---

# Roadmap

## Phase 1 — Historical foundation

- [x] TWIC ingestion
- [x] Feature engineering
- [x] Baseline expert
- [x] ECO expert
- [x] Momentum expert
- [x] ACPL research specialist
- [x] OOF prediction pipeline
- [x] Meta learning

## Phase 2 — Learning and governance

- [x] Prediction ledger
- [x] Surprise scoring
- [x] Experience Replay
- [x] Model registry
- [x] Champion/Challenger evaluation
- [x] Minimum-improvement gate
- [x] Anomaly investigation

## Phase 3 — Live chess

- [x] Normalized chess events
- [x] Tournament configuration
- [x] Chess-Results adapter
- [x] T0 pregame inference
- [x] Persistent ledger integration
- [x] Worker restart/recovery
- [x] Result resolution
- [ ] Live move/ECO adapter
- [ ] T1 worker integration
- [ ] End-to-end live tournament test

## Phase 4 — Tournament intelligence

- [ ] Current standings ingestion
- [ ] Game-level remaining-round forecast
- [ ] Expected final standings
- [ ] Position/top-N probabilities
- [ ] Board-point / match-point forecasting
- [ ] Full Monte Carlo tournament simulation

## Phase 5 — Product layer

- [ ] Streamlit live dashboard
- [ ] Calibration dashboard
- [ ] Prediction history
- [ ] Surprise/anomaly views
- [ ] Model governance views

## Phase 6 — Self-improving production system

- [ ] Scheduled retraining
- [ ] Automated challenger generation
- [ ] Automated holdout evaluation
- [ ] Controlled model deployment
- [ ] Continuous monitoring
- [ ] Cross-tournament validation

## Phase 7 — Beyond chess

The long-term architecture is intended to generalize the same forecasting lifecycle to other sports:

```text
Sport Event
    ↓
Normalized Event
    ↓
Pregame Forecast
    ↓
Live Information Updates
    ↓
Outcome
    ↓
Evaluation
    ↓
Experience Replay
    ↓
Governed Model Improvement
```

Chess is the first domain because it provides a rich combination of historical data, structured events, live state changes, and measurable outcomes.


# Project Philosophy

Blunder Buffer is ultimately an experiment in **self-improving probabilistic forecasting**.

The central question is not simply:

> "Can the model predict the next chess game?"

It is:

> **"Can a forecasting system observe its own prediction errors, preserve what it already knows, learn from important failures, and improve without promoting itself blindly?"**

The tournament layer extends that question from individual games to larger decisions:

```text
Game Forecast
      ↓
Round Forecast
      ↓
Standings Forecast
      ↓
Tournament Simulation
```


---

# License

No license has been specified yet.
