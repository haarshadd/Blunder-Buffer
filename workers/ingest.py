import requests
import sqlite3
import zipfile
import io
import os
import chess.pgn
from datetime import datetime

# --- CONFIGURATION ---
# The most recent TWIC issue as of July 2026 is 1652.
TARGET_TWIC_NUM = 1652 
# TWIC uses strict FIDE naming conventions (Lastname, Firstname)
TARGET_PLAYERS = ["Carlsen", "Caruana", "Gukesh", "Nakamura", "Firouzja", "Ding"]
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "omni_pundit.db")

def setup_database():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chess_games (
            id TEXT PRIMARY KEY,
            white_player TEXT,
            black_player TEXT,
            winner TEXT,
            opening TEXT,
            pgn TEXT,
            fetched_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def fetch_and_filter_twic(twic_num):
    print(f"[{datetime.now()}] Downloading TWIC Issue #{twic_num}...")
    url = f"https://theweekinchess.com/zips/twic{twic_num}g.zip"
    
    headers = {'User-Agent': 'OmniPundit-StudentProject/1.0'}
    
    try:
        # timeout=(connect, read) - without this, a stalled or slow TWIC
        # server hangs the whole pipeline indefinitely with no error. 30s
        # read timeout is generous for a multi-MB zip on a normal connection.
        response = requests.get(url, headers=headers, timeout=(10, 30))
        response.raise_for_status()
        
        # Open the ZIP file entirely in memory (no messy desktop files!)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the .pgn file inside the zip
            pgn_filename = [name for name in z.namelist() if name.endswith('.pgn')][0]
            
            with z.open(pgn_filename) as pgn_file:
                # Convert the byte stream to a string stream for the chess library
                pgn_text = io.TextIOWrapper(pgn_file, encoding='utf-8')
                return filter_games(pgn_text, twic_num)
                
    except Exception as e:
        print(f"Error fetching TWIC {twic_num}: {e}")
        return []

def filter_games(pgn_stream, twic_num):
    """Scans the massive PGN file and only keeps games featuring our target players."""
    print(f"[{datetime.now()}] Scanning thousands of games for target players...")
    saved_games = []
    
    while True:
        # read_game() reads one game at a time, preventing memory crashes
        game = chess.pgn.read_game(pgn_stream)
        if game is None:
            break # End of file
            
        white = game.headers.get("White", "")
        black = game.headers.get("Black", "")
        
        # TWIC uses "Lastname, Firstname" - compare the exact last-name token
        # instead of substring matching. Substring matching (e.g. "Ding" in
        # black) silently matched unrelated players like "Dingemans", which
        # would have quietly polluted the dataset with the wrong games.
        white_lastname = white.split(",")[0].strip()
        black_lastname = black.split(",")[0].strip()
        
        # If either player is in our VIP list, keep the game
        if white_lastname in TARGET_PLAYERS or black_lastname in TARGET_PLAYERS:
            saved_games.append({
                # Generate a unique ID using the TWIC number and players
                'id': f"twic_{twic_num}_{white}_{black}_{game.headers.get('Date', 'unknown')}".replace(" ", ""),
                'white': white,
                'black': black,
                'winner': game.headers.get("Result", "*"),
                'opening': game.headers.get("ECO", "Unknown"),
                # Exporter converts the game object back to raw text for SQLite
                'pgn': str(game) 
            })
            
    return saved_games

def save_games_to_db(games):
    if not games:
        print(f"[{datetime.now()}] Pipeline finished. No relevant games found in this batch.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inserted_count = 0
    
    for game in games:
        fetched_at = datetime.now().isoformat()
        cursor.execute('''
            INSERT OR IGNORE INTO chess_games (id, white_player, black_player, winner, opening, pgn, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (game['id'], game['white'], game['black'], game['winner'], game['opening'], game['pgn'], fetched_at))
        
        if cursor.rowcount == 1:
            inserted_count += 1
            
    conn.commit()
    conn.close()
    print(f"[{datetime.now()}] Pipeline finished. Captured {inserted_count} brand-new official matches.")

if __name__ == "__main__":
    setup_database()
    print("Omni-Pundit TWIC worker running...")
    
    # Run it for the most recent week
    games = fetch_and_filter_twic(TARGET_TWIC_NUM)
    save_games_to_db(games)