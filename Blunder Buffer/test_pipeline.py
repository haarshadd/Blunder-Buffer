# test_pipeline.py (in root folder)
from core.replay_buffer import build_training_batch
from core.registry import init_registry
from core.orchestrator import evaluate_and_promote
import pandas as pd
import os

# Bulletproof absolute pathing:
# Goes from root/test_pipeline.py -> into data/omni_pundit.db
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "omni_pundit.db"))

def test_system():
    init_registry(DB_PATH)

    print("--- 1. Testing Scoped Replay Buffer ---")
    batch = build_training_batch(
        db_path=DB_PATH,
        sport='chess',
        model_version_id='baseline_v2', # Strictly scoping to Baseline errors
        total_size=10000,
        since_iso='2025-01-01',          
        historical_cutoff_iso='2018-01-01' 
    )

    df = pd.DataFrame(batch)
    print(f"Successfully sampled {len(batch)} games!")
    print(f"Highest Baseline Surprise Score: {df['surprise_score'].max():.4f}\n")
    
    print("--- 2. Testing Orchestrator Noise Margin ---")
    # Simulating a challenger that improves, but only by 0.0001 (noise limit)
    champ_metrics = {'fresh_holdout_log_loss': 0.8420}
    chal_metrics = {'fresh_holdout_log_loss': 0.8419} 
    
    success, reason = evaluate_and_promote(
        db_path=DB_PATH,
        sport='chess',
        layer='meta',
        challenger_id='meta_wide_deep_v3_test',
        training_window={"start": "2026-01-01", "end": "2026-07-01"},
        challenger_metrics=chal_metrics,
        champion_metrics=champ_metrics,
        min_improvement=0.002  # Requires at least a 0.002 jump to promote
    )

if __name__ == "__main__":
    test_system()