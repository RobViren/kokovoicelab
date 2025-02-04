import argparse
import sqlite3
import numpy as np
import soundfile as sf
from pathlib import Path
from kokoro_onnx import Kokoro
import io

def convert_array(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array"""
    out = io.BytesIO(blob)
    return np.load(out)

def get_voice_vector(conn: sqlite3.Connection, voice_name: str) -> np.ndarray:
    """Get style vector for a specific voice name"""
    # Register the numpy array converter
    sqlite3.register_converter("array", convert_array)
    
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM voices WHERE name = ?", (voice_name,))
    voice = cursor.fetchone()
    
    if not voice:
        raise ValueError(f"No voice found with name: {voice_name}")
    
    # Print voice details
    print(f"\nVoice details:")
    print(f"Name: {voice[0]}, Gender: {voice[1]}, Language: {voice[2]}, Quality: {voice[3]}")
    
    return voice[5]  # style_vector is at index 5

def main():
    parser = argparse.ArgumentParser(description='Generate audio using a specific voice from the database')
    parser.add_argument('--text', required=True,
                      help='Text to synthesize')
    parser.add_argument('--voice-name', required=True,
                      help='Name of the voice to use')
    parser.add_argument('--speed', type=float, default=1.0,
                      help='Speech speed (default: 1.0)')
    parser.add_argument('--output-dir', default='output',
                      help='Output directory for audio file')
    parser.add_argument('--db-path', default='voices.db',
                      help='Path to SQLite database')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(args.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    
    try:
        # Get voice vector
        voice_style = get_voice_vector(conn, args.voice_name)
        
        # Initialize Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        
        # Generate audio
        print(f"\nGenerating audio for voice: {args.voice_name}")
        samples, sample_rate = kokoro.create(
            args.text,
            voice=voice_style,
            speed=args.speed,
            lang='en-us'  # You might want to make this configurable
        )
        
        # Save audio file
        output_file = output_dir / f"{args.voice_name}_{args.text[:20]}.wav"
        sf.write(output_file, samples, sample_rate)
        print(f"Created audio file: {output_file}")
                
    finally:
        conn.close()

if __name__ == "__main__":
    main() 