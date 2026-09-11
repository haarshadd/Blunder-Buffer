# investigate_anomaly.py
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "omni_pundit.db"))

def find_asymptote_errors():
    print("--- Investigating 20.7233 Surprise Score Anomalies ---\n")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT 
            p.event_id, 
            p.model_version_id, 
            p.actual_outcome, 
            p.predicted_probs, 
            b.white, 
            b.black, 
            b.white_elo, 
            b.black_elo, 
            b.elo_diff,
            b.has_elo
        FROM predictions_ledger p
        JOIN baseline_features b ON p.event_id = b.game_id
        WHERE p.surprise_score >= 20.7
        ORDER BY b.elo_diff DESC
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    if len(df) == 0:
        print("No extreme anomalies found.")
        return
        
    print(f"Found {len(df)} predictions hitting the 1e-9 probability ceiling.")
    print("Sample of anomalous data rows:")
    
    # Adjust pandas display options to print cleanly to the terminal
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.head(10))

if __name__ == "__main__":
    find_asymptote_errors()