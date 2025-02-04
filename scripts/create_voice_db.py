import sqlite3
import numpy as np
from kokoro_onnx import Kokoro
import json
from pathlib import Path
from typing import Dict
import io

def adapt_array(arr: np.ndarray) -> bytes:
    """Convert numpy array to binary for SQLite storage"""
    out = io.BytesIO()
    np.save(out, arr)
    return out.getvalue()

def convert_array(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array"""
    out = io.BytesIO(blob)
    return np.load(out)

# Register the adapters with SQLite
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def create_voice_database(db_path: str = "voices.db"):
    """Create and initialize the voice database"""
    # Connect to SQLite database with numpy array support
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # Create the voices table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS voices (
        name TEXT PRIMARY KEY,
        gender TEXT NOT NULL,
        language TEXT NOT NULL,
        quality INTEGER NOT NULL,
        training_duration TEXT,
        style_vector array NOT NULL,
        is_synthetic BOOLEAN NOT NULL DEFAULT 0,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    return conn

def load_voice_data(file_path: str = "voice-data.json") -> Dict:
    """Load voice metadata from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def populate_database(conn: sqlite3.Connection, voice_data: Dict, kokoro: Kokoro):
    """Populate the database with voice data and style vectors"""
    cursor = conn.cursor()
    
    for voice in voice_data['voices']:
        try:
            # Get style vector for the voice
            style_vector = kokoro.get_voice_style(voice['name'])
            
            # Insert voice data and style vector into database
            cursor.execute('''
            INSERT OR REPLACE INTO voices 
            (name, gender, language, quality, training_duration, style_vector, is_synthetic, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                voice['name'],
                voice['gender'],
                voice['language'],
                voice['quality'],
                voice.get('training_duration'),
                style_vector,
                voice.get('is_synthetic', False),  # Default to False if not specified
                voice.get('notes')  # Will be None if not specified
            ))
            
            print(f"Added voice: {voice['name']}")
            
        except Exception as e:
            print(f"Error processing voice {voice['name']}: {e}")
    
    conn.commit()

def main():
    # Initialize Kokoro
    print("Initializing Kokoro...")
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    
    # Load voice metadata
    print("Loading voice metadata...")
    voice_data = load_voice_data()
    
    # Create database
    print("Creating database...")
    db_path = "voices.db"
    conn = create_voice_database(db_path)
    
    # Populate database
    print("Populating database with voices...")
    populate_database(conn, voice_data, kokoro)
    
    # Verify the data
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM voices")
    count = cursor.fetchone()[0]
    print(f"\nDatabase created successfully with {count} voices")
    
    # Example query to verify data
    cursor.execute("""
    SELECT name, gender, language, quality 
    FROM voices 
    ORDER BY quality DESC 
    LIMIT 5
    """)
    print("\nTop 5 quality voices:")
    for row in cursor.fetchall():
        print(f"Name: {row[0]}, Gender: {row[1]}, Language: {row[2]}, Quality: {row[3]}")
    
    conn.close()

if __name__ == "__main__":
    main() 