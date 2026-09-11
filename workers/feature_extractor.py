import sqlite3
import chess.pgn
import io
import os
import sys
import pandas as pd
import numpy as np

# Force Python to print everything to the terminal instantly without buffering
sys.stdout.reconfigure(line_buffering=True)

# Bulletproof absolute pathing
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "omni_pundit.db"))

def setup_feature_table():
    """Creates a clean, ML-ready tabular database table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baseline_features (
            game_id TEXT PRIMARY KEY,
            date DATE,
            white TEXT,
            black TEXT,
            white_elo INTEGER,
            black_elo INTEGER,
            elo_diff INTEGER,
            has_elo INTEGER,
            white_historical_wins INTEGER,
            black_historical_wins INTEGER,
            historical_draws INTEGER,
            target_result INTEGER
        )
    ''')
    
    try:
        cursor.execute("ALTER TABLE baseline_features ADD COLUMN has_elo INTEGER")
    except sqlite3.OperationalError:
        pass
        
    # Dynamically add the psychological context columns
    new_cols = ['w_rest', 'b_rest', 'w_fatigue', 'b_fatigue', 'w_tilt', 'b_tilt', 'black_draw_rate_10']
    for col in new_cols:
        try:
            cursor.execute(f"ALTER TABLE baseline_features ADD COLUMN {col} REAL")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

def extract_features():
    print("--- Starting Omni-Pundit Feature Extraction Engine ---")
    print(f"Target Database: {os.path.abspath(DB_PATH)}")
    
    if not os.path.exists(DB_PATH):
        print("ERROR: Database file not found! Please run the ingestion script first.")
        return
        
    setup_feature_table()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM chess_games WHERE pgn IS NOT NULL")
    raw_count = cursor.fetchone()[0]
    print(f"Found {raw_count} raw games in 'chess_games' table ready for extraction.")
    
    if raw_count == 0:
        print("Exiting: No games found to extract features from.")
        conn.close()
        return
        
    cursor.execute("SELECT id, pgn FROM chess_games WHERE pgn IS NOT NULL")
    raw_games = cursor.fetchall()
    
    parsed_data = []
    
    for game_id, pgn_text in raw_games:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None: 
            continue
        
        try:
            w_elo = int(game.headers.get("WhiteElo", 0) or 0)
        except ValueError:
            w_elo = 0
        try:
            b_elo = int(game.headers.get("BlackElo", 0) or 0)
        except ValueError:
            b_elo = 0

        has_elo = 1 if (w_elo > 0 and b_elo > 0) else 0
            
        date_played = game.headers.get("Date", "1970.01.01").replace(".", "-")
        white = game.headers.get("White", "Unknown")
        black = game.headers.get("Black", "Unknown")
        
        white_key = white.replace(" ", "").strip().lower()
        black_key = black.replace(" ", "").strip().lower()
        
        res_str = game.headers.get("Result", "*")
        if res_str == "1-0": target = 1
        elif res_str == "0-1": target = -1
        elif res_str == "1/2-1/2": target = 0
        else: continue 
            
        parsed_data.append({
            'id': game_id, 'date': date_played, 
            'white': white, 'black': black, 
            'white_key': white_key, 'black_key': black_key,
            'w_elo': w_elo, 'b_elo': b_elo, 
            'elo_diff': w_elo - b_elo,
            'has_elo': has_elo,
            'target': target
        })

    parsed_data.sort(key=lambda x: x['date'])
    
    h2h_tracker = {}
    features_to_save = []
    
    for match in parsed_data:
        p1, p2 = sorted([match['white_key'], match['black_key']])
        matchup_key = f"{p1}|{p2}"
        
        if matchup_key not in h2h_tracker:
            h2h_tracker[matchup_key] = {'p1_wins': 0, 'p2_wins': 0, 'draws': 0}
            
        history = h2h_tracker[matchup_key]
        
        if match['white_key'] == p1:
            w_hist_wins = history['p1_wins']
            b_hist_wins = history['p2_wins']
        else:
            w_hist_wins = history['p2_wins']
            b_hist_wins = history['p1_wins']
            
        draws = history['draws']
        
        features_to_save.append((
            match['id'], match['date'], match['white'], match['black'],
            match['w_elo'], match['b_elo'], match['elo_diff'], match['has_elo'],
            w_hist_wins, b_hist_wins, draws, match['target']
        ))

        if match['target'] == 1: 
            if match['white_key'] == p1: history['p1_wins'] += 1
            else: history['p2_wins'] += 1
        elif match['target'] == -1: 
            if match['black_key'] == p1: history['p1_wins'] += 1
            else: history['p2_wins'] += 1
        else:
            history['draws'] += 1

    # <--- OUTDENT THIS SECTION SO IT IS OUTSIDE THE 'for match in parsed_data:' LOOP
    cursor.executemany('''
            INSERT OR REPLACE INTO baseline_features (
                game_id, date, white, black, white_elo, black_elo, 
                elo_diff, has_elo, white_historical_wins, 
                black_historical_wins, historical_draws, target_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', features_to_save)
    
    conn.commit()
    conn.close()
    print(f"Extraction Complete! Successfully saved {len(features_to_save)} rows into 'baseline_features'.")


def compute_psych_context():
    """Calculates and materially saves Tilt, Fatigue, Rest, and Draw Rates into the database."""
    print("--- Computing Psychological Context Features ---")
    conn = sqlite3.connect(DB_PATH)
    
    query = """
        SELECT game_id, date, white, black, target_result 
        FROM baseline_features 
        ORDER BY date ASC
    """
    df = pd.read_sql(query, conn)
    
    if len(df) == 0:
        print("No games found. Skipping context computation.")
        conn.close()
        return

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    print("Engineering Psychological Context Vectors (Tilt, Fatigue, Rest)...")
    player_timeline = []
    
    df['white_key'] = df['white'].str.replace(" ", "").str.strip().str.lower()
    df['black_key'] = df['black'].str.replace(" ", "").str.strip().str.lower()
    
    for row in df.itertuples():
        player_timeline.append({
            'game_id': row.game_id, 'date': row.date, 'player': row.white_key, 
            'color': 'white', 'result_class': 1 if row.target_result == 1 else (0 if row.target_result == 0 else -1)
        })
        player_timeline.append({
            'game_id': row.game_id, 'date': row.date, 'player': row.black_key, 
            'color': 'black', 'result_class': 1 if row.target_result == -1 else (0 if row.target_result == 0 else -1)
        })

    pdf = pd.DataFrame(player_timeline).sort_values(['player', 'date']).copy()

    # 1. Rest Days
    pdf['prev_date'] = pdf.groupby('player')['date'].shift(1)
    pdf['rest_days'] = (pdf['date'] - pdf['prev_date']).dt.days.fillna(30)
    pdf['rest_days'] = np.clip(pdf['rest_days'], 0, 90)

    # 2. Fatigue (7D Rolling)
    pdf = pdf.set_index('date')
    fatigue_values = pdf.groupby('player')['game_id'].rolling('7D').count().values - 1
    pdf['fatigue_7d'] = fatigue_values
    pdf = pdf.reset_index()

    # 3. Tilt (Consecutive Losses Only)
    pdf['loss_flag'] = (pdf['result_class'] == -1).astype(int)
    pdf['tilt_streak'] = pdf.groupby('player')['loss_flag'].transform(
        lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)
    )
    pdf['tilt_streak'] = pdf.groupby('player')['tilt_streak'].shift(1).fillna(0)

    # 4. Black Draw Rate
    pdf['draw_flag'] = (pdf['result_class'] == 0).astype(int)
    pdf['black_draw_flag'] = np.where(pdf['color'] == 'black', pdf['draw_flag'], np.nan)
    pdf['black_draw_rate_10'] = pdf.groupby('player')['black_draw_flag'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
    )
    pdf['black_draw_rate_10'] = pdf['black_draw_rate_10'].fillna(0.0)

    pdf = pdf.sort_values('date')

    # Merge Context Features back to Game IDs
    w_context = pdf[pdf['color'] == 'white'][['game_id', 'rest_days', 'fatigue_7d', 'tilt_streak']].rename(
        columns={'rest_days': 'w_rest', 'fatigue_7d': 'w_fatigue', 'tilt_streak': 'w_tilt'}
    )
    b_context = pdf[pdf['color'] == 'black'][['game_id', 'rest_days', 'fatigue_7d', 'tilt_streak', 'black_draw_rate_10']].rename(
        columns={'rest_days': 'b_rest', 'fatigue_7d': 'b_fatigue', 'tilt_streak': 'b_tilt'}
    )

    df_update = w_context.merge(b_context, on='game_id', how='inner')
    df_update = df_update.drop_duplicates(subset=['game_id'])

    print(f"Committing {len(df_update)} psychological context vectors to database...")
    update_data = list(df_update[['w_rest', 'b_rest', 'w_fatigue', 'b_fatigue', 'w_tilt', 'b_tilt', 'black_draw_rate_10', 'game_id']].itertuples(index=False, name=None))
    
    cursor = conn.cursor()
    cursor.executemany('''
        UPDATE baseline_features 
        SET w_rest = ?, b_rest = ?, w_fatigue = ?, b_fatigue = ?, w_tilt = ?, b_tilt = ?, black_draw_rate_10 = ?
        WHERE game_id = ?
    ''', update_data)
    
    conn.commit()
    conn.close()
    print("Psychological Context Materialization Complete!")
    print("--- End of Feature Extraction Pipeline ---")

if __name__ == "__main__":
    extract_features()
    compute_psych_context()