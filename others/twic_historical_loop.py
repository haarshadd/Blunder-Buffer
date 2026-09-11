import requests
import sqlite3
import zipfile
import io
import os
import chess.pgn
from datetime import datetime
import time

# --- CONFIGURATION ---
START_TWIC = 920   # Early 2010
END_TWIC = 1651    # Up to the week before the one we already grabbed
# Instead of a hardcoded name list, we use the historical Top 100 rating cutoff.
# Since 2010, the threshold to enter the Top 100 has hovered around 2640-2655.
# Filtering by 2640 guarantees we capture the games of elite players dynamically.
MIN_ELO_THRESHOLD = 2640 
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "omni_pundit.db")

def filter_games(pgn_stream, twic_num):
    saved_games = []
    while True:
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            break 
            
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        
        # Safely parse Elos (defaulting to 0 if unrated or missing from the broadcast)
        try:
            white_elo = int(game.headers.get("WhiteElo", "0") or "0")
        except ValueError:
            white_elo = 0
            
        try:
            black_elo = int(game.headers.get("BlackElo", "0") or "0")
        except ValueError:
            black_elo = 0
        
        # If EITHER player is at a Top 100 level, we save the game
        if white_elo >= MIN_ELO_THRESHOLD or black_elo >= MIN_ELO_THRESHOLD:
            saved_games.append({
                'id': f"twic_{twic_num}_{white}_{black}_{game.headers.get('Date', 'unknown')}".replace(" ", ""),
                'white': white,
                'black': black,
                'winner': game.headers.get("Result", "*"),
                'opening': game.headers.get("ECO", "Unknown"),
                'pgn': str(game) 
            })
    return saved_games

def fetch_and_save_twic(twic_num):
    url = f"https://theweekinchess.com/zips/twic{twic_num}g.zip"
    headers = {'User-Agent': 'OmniPundit-HistoricalLoader/1.0'}
    
    try:
        response = requests.get(url, headers=headers, timeout=(10, 30))
        if response.status_code != 200:
            print(f"[{datetime.now()}] Skipping TWIC {twic_num} (HTTP {response.status_code})")
            return
            
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            pgn_filename = [name for name in z.namelist() if name.endswith('.pgn')][0]
            with z.open(pgn_filename) as pgn_file:
                pgn_text = io.TextIOWrapper(pgn_file, encoding='utf-8')
                games = filter_games(pgn_text, twic_num)
                
        if games:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            inserted_count = 0
            for game in games:
                cursor.execute('''
                    INSERT OR IGNORE INTO chess_games (id, white_player, black_player, winner, opening, pgn, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (game['id'], game['white'], game['black'], game['winner'], game['opening'], game['pgn'], datetime.now().isoformat()))
                if cursor.rowcount == 1:
                    inserted_count += 1
            conn.commit()
            conn.close()
            print(f"[{datetime.now()}] TWIC {twic_num} -> Saved {inserted_count} elite games.")
        else:
            print(f"[{datetime.now()}] TWIC {twic_num} -> 0 target games found.")
            
    except Exception as e:
        print(f"[{datetime.now()}] Error on TWIC {twic_num}: {e}")

if __name__ == "__main__":
    print(f"--- Starting Massive Historical Data Load (Issues {START_TWIC} to {END_TWIC}) ---")
    print("This will take a while. You can leave it running in the background.")
    
    for current_twic in range(START_TWIC, END_TWIC + 1):
        fetch_and_save_twic(current_twic)
        time.sleep(2.5) # BE POLITE TO TWIC SERVERS! 
        
    print("Historical load complete! You now have a 16-year baseline database.")