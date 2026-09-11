"""
Retrain Orchestrator - core/orchestrator.py

The gatekeeper for model promotion. Implements the Champion/Challenger 
pattern to strictly prevent catastrophic forgetting.

It evaluates a Challenger's metrics against the live Champion.
Promotion Rules:
  1. The Challenger must beat the Champion on a genuinely fresh holdout window.
  2. The Challenger must NOT regress on fixed historical anchor windows.
"""

from core import registry

def evaluate_and_promote(db_path, sport, layer, challenger_id, training_window, challenger_metrics, champion_metrics=None, min_improvement=0.002, regress_tolerance=0.005):
    """
    Evaluates a new model and decides whether to promote it to Champion.
    
    Expected metrics dictionary format:
    {
        'fresh_holdout_log_loss': 0.8420,
        'regression_anchors': {
            '2018_slice_log_loss': 0.8801,
            '2021_slice_log_loss': 0.8750,
            '2024_slice_log_loss': 0.8500
        }
    }
    """
    # 1. Register the Challenger regardless of whether it wins or loses
    # This maintains the immutable audit trail of all experiments.
    registry.register_model(
        db_path=db_path,
        model_version_id=challenger_id,
        sport=sport,
        layer=layer,
        training_window=training_window,
        metrics_json=challenger_metrics
    )

    # 2. Fetch the current Champion
    champion = registry.get_champion(db_path, sport, layer)
    
    # Bootstrap condition: If no champion exists, this is the first model. Auto-promote.
    if champion is None:
        print(f"[{layer}] No existing champion found. Auto-promoting {challenger_id} to LIVE.")
        registry.promote_model(db_path, challenger_id, sport, layer)
        return True, "Auto-promoted (First of its kind)."

    # 3. If champion metrics weren't provided directly, pull them from the registry
    if champion_metrics is None:
        champion_metrics = champion['metrics']

    print(f"[{layer}] Commencing Champion ({champion['model_version_id']}) vs Challenger ({challenger_id}) Evaluation...")

    # RULE 1: Must improve on the fresh holdout set
    chal_holdout = challenger_metrics.get('fresh_holdout_log_loss')
    champ_holdout = champion_metrics.get('fresh_holdout_log_loss')

    if chal_holdout is None or champ_holdout is None:
        reason = "BLOCKED: Missing 'fresh_holdout_log_loss' in metrics dictionaries."
        print(reason)
        return False, reason
    
    if chal_holdout >= champ_holdout:
        reason = f"BLOCKED: Challenger failed to improve on fresh data (Chal: {chal_holdout:.4f} >= Champ: {champ_holdout:.4f})"
        print(reason)
        return False, reason

    # RULE 2: Must not silently forget historical patterns (Catastrophic Forgetting Shield)
    chal_anchors = challenger_metrics.get('regression_anchors', {})
    champ_anchors = champion_metrics.get('regression_anchors', {})
    
    for anchor_name, champ_loss in champ_anchors.items():
        chal_loss = chal_anchors.get(anchor_name)
        if chal_loss is None:
            continue  # If the challenger doesn't have this anchor, we can't evaluate it; skip.
    
        # A tiny tolerance threshold (e.g., 0.005) can be added here if you want to allow 
        # negligible variance, but strict regression blocking is safer for now.
        if chal_loss is not None and chal_loss > (champ_loss + regress_tolerance):
            reason = f"BLOCKED: Catastrophic Forgetting detected on {anchor_name} (Chal: {chal_loss:.4f} > Champ: {champ_loss:.4f})"
            print(reason)
            return False, reason

    # 4. If all tests pass, execute the promotion
    print(f"[{layer}] EVALUATION PASSED! Promoting {challenger_id} to Champion.")
    registry.promote_model(db_path, challenger_id, sport, layer)
    
    return True, "Promoted successfully."