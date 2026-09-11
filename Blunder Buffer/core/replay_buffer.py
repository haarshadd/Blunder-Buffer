"""
Experience Replay Sampler - core/replay_buffer.py

Implements a dynamic sampling strategy over the predictions_ledger. 
Yields a mixed batch of data for retraining to mathematically prevent 
catastrophic forgetting and ensure stable calibration.

Mixes 3 slices:
  1. Prioritized: Top-K highest surprise scores (confident mistakes).
  2. Recent Random: Uniform sample of recent predictions (maintains calibration).
  3. Historical Anchor: Uniform sample from deep history (prevents forgetting).
"""

from core import ledger
import random

def build_training_batch(db_path, sport, model_version_id, total_size, since_iso, historical_cutoff_iso, ratios=None):
    """
    Constructs a training batch based on the defined ratio logic.
    
    :param total_size: The total number of rows you want in the training batch.
    :param since_iso: The timestamp defining "recent" data (e.g., since last retrain).
    :param historical_cutoff_iso: The timestamp defining "deep history".
    :param ratios: Dictionary containing the sampling mix.
    """
    if ratios is None:
        ratios = {
            'prioritized': 0.4,
            'recent_random': 0.4,
            'historical_anchor': 0.2
        }
        
    # Calculate target sizes for each slice
    size_prio = int(total_size * ratios['prioritized'])
    size_recent = int(total_size * ratios['recent_random'])
    size_hist = total_size - size_prio - size_recent # Absorb rounding remainder

    # Fetch slices directly from the ledger using the read helpers
    prioritized_slice = ledger.get_top_surprise(
        db_path=db_path, 
        k=size_prio, 
        since_iso=since_iso, 
        sport=sport,
        model_version_id=model_version_id  # Optional: could be used to filter by specific model version
    )
    
    recent_slice = ledger.get_random_sample(
        db_path=db_path, 
        n=size_recent, 
        since_iso=since_iso, 
        sport=sport,
        model_version_id= model_version_id  # Optional: could be used to filter by specific model version
    )
    
    historical_slice = ledger.get_historical_anchor_sample(
        db_path=db_path, 
        n=size_hist, 
        before_iso=historical_cutoff_iso, 
        sport=sport,
        model_version_id=model_version_id  # Optional: could be used to filter by specific model version
    )
    
    # Combine the slices into a single replay batch
    combined_batch = prioritized_slice + recent_slice + historical_slice
    
    # Optional: Deduplicate by event_id in case the 'recent random' slice 
    # accidentally grabbed an event that was already in the 'prioritized' slice.
    unique_batch = {row['event_id']: row for row in combined_batch}.values()
    
    # Shuffle the final batch so the model doesn't learn in contiguous blocks 
    # of "all mistakes" followed by "all old games".
    final_list = list(unique_batch)
    random.shuffle(final_list)
    
    return final_list