import argparse
import sqlite3
import numpy as np
import torch
import io
from pathlib import Path

def convert_array(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array"""
    out = io.BytesIO(blob)
    return np.load(out)

def get_voice_vector(conn: sqlite3.Connection, voice_name: str) -> np.ndarray:
    """Get style vector for a specific voice name"""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM voices WHERE name = ?", (voice_name,))
    voice = cursor.fetchone()
    
    if not voice:
        raise ValueError(f"No voice found with name: {voice_name}")
    
    # Print voice details
    print(f"\nVoice details:")
    print(f"Name: {voice[0]}")
    print(f"Gender: {voice[1]}")
    print(f"Language: {voice[2]}")
    print(f"Quality: {voice[3]}")
    if voice[7]:  # notes field
        print(f"Notes: {voice[7]}")
    
    return voice[5]  # style_vector is at index 5

def export_all_voices(conn: sqlite3.Connection, output_file: Path) -> None:
    """Export all voices from database to a single .bin file"""
    cursor = conn.cursor()
    cursor.execute("SELECT name, style_vector FROM voices")
    voices = cursor.fetchall()
    
    if not voices:
        raise ValueError("No voices found in database")
    
    # Create dictionary of voice_name: style_vector pairs
    voice_dict = {name: style_vector for name, style_vector in voices}
    print(f"\nExporting {len(voice_dict)} voices to {output_file}")
    
    # Save as NPZ file
    with open(output_file, "wb") as f:
        np.savez(f, **voice_dict)
    
    print(f"Successfully exported voices to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Export voice(s) from the database')
    parser.add_argument('--voice-name',
                      help='Name of the voice to export (if not specified, exports all voices)')
    parser.add_argument('--output-dir', default='exported_voices',
                      help='Output directory for the .pt file(s)')
    parser.add_argument('--db-path', default='voices.db',
                      help='Path to SQLite database')
    parser.add_argument('--export-all', action='store_true',
                      help='Export all voices to a single voices.bin file')
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Connect to database with numpy array support
    sqlite3.register_converter("array", convert_array)
    conn = sqlite3.connect(args.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    
    try:
        if args.export_all:
            # Export all voices to a single .bin file
            output_file = output_dir / "voices.bin"
            export_all_voices(conn, output_file)
        elif args.voice_name:
            # Export single voice as .pt file
            voice_vector = get_voice_vector(conn, args.voice_name)
            
            # Convert numpy array to PyTorch tensor
            tensor = torch.from_numpy(voice_vector)
            
            # Save as .pt file
            output_file = output_dir / f"{args.voice_name}.pt"
            torch.save(tensor, output_file)
            print(f"\nExported voice to: {output_file}")
        else:
            parser.error("Either --voice-name or --export-all must be specified")
                
    finally:
        conn.close()

if __name__ == "__main__":
    main() 