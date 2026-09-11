import sqlite3
import chess
import chess.engine
import chess.pgn
import io
import os
import sys
import time
from datetime import datetime

# Force unbuffered output for real-time terminal printing
sys.stdout.reconfigure(line_buffering=True)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "omni_pundit.db")
STOCKFISH_EXECUTABLE = "stockfish.exe" if os.name == 'nt' else "stockfish"
STOCKFISH_PATH = os.path.join(os.path.dirname(__file__), STOCKFISH_EXECUTABLE)

# --- Hardware Optimization Settings ---
MAX_CP_LOSS_PER_MOVE = 1000
ANALYSIS_DEPTH = 14
BATCH_SIZE = 5       
MAX_GAMES_CAP = 10000 

def setup_database_schema():
    """Ensures the database has the columns and our tracking flag layout."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Core ACPL columns
    try:
        cursor.execute("ALTER TABLE chess_games ADD COLUMN white_acpl INTEGER")
        cursor.execute("ALTER TABLE chess_games ADD COLUMN black_acpl INTEGER")
    except sqlite3.OperationalError:
        pass
        
    # Stratified target flag column
    try:
        cursor.execute("ALTER TABLE chess_games ADD COLUMN is_target_acpl INTEGER DEFAULT 0")
        print("Initialized stratified tracking layout in database.")
    except sqlite3.OperationalError:
        pass
        
    conn.commit()
    conn.close()

def tag_stratified_targets():
    """Selects ~600 random games per year across the entire history to hit the 10k target uniform pool."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if we have already tagged our uniform pool on a previous run
    cursor.execute("SELECT COUNT(*) FROM chess_games WHERE is_target_acpl = 1")
    already_tagged = cursor.fetchone()[0]
    
    if already_tagged > 0:
        print(f"Persistent Pool Found: {already_tagged:,} games are locked in for uniform sampling.")
        conn.close()
        return

    print("Building Uniform Stratified Sample across all historical eras...")
    
    # Extract unique years using the pre-existing baseline features table
    cursor.execute("SELECT DISTINCT SUBSTR(date, 1, 4) FROM baseline_features WHERE date IS NOT NULL")
    years = [row[0] for row in cursor.fetchall() if row[0].isdigit()]
    
    if not years:
        print("ERROR: No historical dates found. Ensure feature_extractor.py ran successfully.")
        conn.close()
        return
        
    target_per_year = MAX_GAMES_CAP // len(years)
    print(f"Detected {len(years)} unique tournament years. Target target: {target_per_year} random games per year.")

    for year in years:
        # Pull random game IDs for the current year
        query = """
            SELECT b.game_id 
            FROM baseline_features b
            JOIN chess_games c ON b.game_id = c.id
            WHERE SUBSTR(b.date, 1, 4) = ?
            ORDER BY RANDOM()
            LIMIT ?
        """
        cursor.execute(query, (year, target_per_year))
        game_ids = [row[0] for row in cursor.fetchall()]
        
        # Tag these specific games as our official ML target pool
        if game_ids:
            cursor.executemany(
                "UPDATE chess_games SET is_target_acpl = 1 WHERE id = ?",
                [(gid,) for gid in game_ids]
            )
            
    conn.commit()
    
    cursor.execute("SELECT COUNT(*) FROM chess_games WHERE is_target_acpl = 1")
    total_tagged = cursor.fetchone()[0]
    print(f"Successfully locked in {total_tagged:,} perfectly uniform games for analysis.")
    conn.close()

def check_progress():
    """Tracks running progress against our tagged 10,000 pool."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM chess_games WHERE is_target_acpl = 1 AND white_acpl IS NOT NULL")
    completed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chess_games WHERE is_target_acpl = 1")
    total_pool = cursor.fetchone()[0]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Uniform Progress: {completed:,} / {total_pool:,} analyzed.")
    
    conn.close()
    return completed >= total_pool

def analyze_game(pgn_text, engine):
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None or game.headers.get("Variant", "Standard") != "Standard":
        return -1, -1
        
    board = game.board()
    white_loss, black_loss = 0, 0
    white_moves, black_moves = 0, 0
    limit = chess.engine.Limit(depth=ANALYSIS_DEPTH)
    
    info = engine.analyse(board, limit)
    prev_eval = info["score"].white().score(mate_score=10000)
    
    for move in game.mainline_moves():
        is_white_turn = board.turn == chess.WHITE
        board.push(move)
        
        info = engine.analyse(board, limit)
        current_eval = info["score"].white().score(mate_score=10000)
        
        if current_eval is not None and prev_eval is not None:
            cp_loss = max(0, prev_eval - current_eval) if is_white_turn else max(0, current_eval - prev_eval)
            cp_loss = min(cp_loss, MAX_CP_LOSS_PER_MOVE)
            
            if is_white_turn:
                white_loss += cp_loss
                white_moves += 1
            else:
                black_loss += cp_loss
                black_moves += 1
                
        prev_eval = current_eval
        
    w_acpl = int(white_loss / white_moves) if white_moves > 0 else 0
    b_acpl = int(black_loss / black_moves) if black_moves > 0 else 0
    return w_acpl, b_acpl

def run_analysis_pipeline():
    print("--- Starting Stratified Omni-Pundit Analysis Engine ---")
    setup_database_schema()
    tag_stratified_targets()
    
    if not os.path.exists(STOCKFISH_PATH):
        print(f"\nERROR: Stockfish binary missing at {STOCKFISH_PATH}!")
        return

    print("Connecting to Stockfish engine...")
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            # Constrained perfectly for HP i3 / 8GB RAM specs
            engine.configure({"Threads": 1, "Hash": 128})
            
            while True:
                if check_progress():
                    print("\n🎯 Uniform 10,000-game dataset completely analyzed! Pipeline successful.")
                    break
                    
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # BLAZING FAST QUERY: Pulls exclusively from our pre-tagged uniform target pool
                cursor.execute(f"""
                    SELECT id, white_player, black_player, pgn 
                    FROM chess_games 
                    WHERE is_target_acpl = 1 AND white_acpl IS NULL 
                    LIMIT {BATCH_SIZE}
                """)
                unprocessed_games = cursor.fetchall()
                conn.close()
                
                if not unprocessed_games:
                    print("Waiting for batch allocations...")
                    time.sleep(10)
                    continue
                
                # Execute engine computations
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                for game_id, white, black, pgn in unprocessed_games:
                    print(f"  -> Processing: {white} vs {black}")
                    w_acpl, b_acpl = analyze_game(pgn, engine)
                    
                    cursor.execute("""
                        UPDATE chess_games 
                        SET white_acpl = ?, black_acpl = ? 
                        WHERE id = ?
                    """, (w_acpl, b_acpl, game_id))
                    conn.commit()
                conn.close()
                
                # Thermal protection cooldown
                print("Batch complete. Cooling CPU down for 10 seconds...")
                time.sleep(10) 
                
    except KeyboardInterrupt:
        print("\n--- Analysis Safely Paused. Progress is securely saved in the database. ---")
    except Exception as e:
        print(f"\nPipeline Exception occurred: {e}")

if __name__ == "__main__":
    run_analysis_pipeline()