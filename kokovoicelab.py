import argparse
import sqlite3
import numpy as np
from kokoro_onnx import Kokoro
import soundfile as sf
from pathlib import Path
from typing import List, Tuple
import io

def convert_array(blob: bytes) -> np.ndarray:
    """Convert binary blob back to numpy array"""
    out = io.BytesIO(blob)
    return np.load(out)

def get_voice_group_vector(conn: sqlite3.Connection, query: str) -> np.ndarray:
    """Execute query and return average style vector for matching voices"""
    # Register the numpy array converter
    sqlite3.register_converter("array", convert_array)
    
    cursor = conn.cursor()
    cursor.execute(query)
    voices = cursor.fetchall()
    
    if not voices:
        raise ValueError(f"No voices found for query: {query}")
    
    # Print matched voices for transparency
    print(f"\nVoices matched by query:")
    for voice in voices:
        print(f"Name: {voice[0]}, Gender: {voice[1]}, Language: {voice[2]}, Quality: {voice[3]}")
    
    # Extract and average style vectors
    style_vectors = [voice[5] for voice in voices]  # style_vector is at index 5
    return np.mean(style_vectors, axis=0)

def interpolate_styles(style1: np.ndarray, style2: np.ndarray, factor: float) -> np.ndarray:
    """Interpolate between two style vectors"""
    diff_vector = style2 - style1
    midpoint = (style1 + style2) / 2
    return midpoint + (diff_vector * factor / 2)

def adapt_array(arr: np.ndarray) -> bytes:
    """Convert numpy array to binary for SQLite storage"""
    out = io.BytesIO()
    np.save(out, arr)
    return out.getvalue()

def main():
    parser = argparse.ArgumentParser(description='Generate voice interpolation samples using SQL queries')
    parser.add_argument('--source-query', required=True, 
                      help='SQL query for selecting source voices')
    parser.add_argument('--target-query', required=True,
                      help='SQL query for selecting target voices')
    parser.add_argument('--text', default="Hello, world!",
                      help='Text to synthesize')
    parser.add_argument('--ranges', default="-2,-1,-0.5,0,0.5,1,2",
                      help='Comma-separated list of interpolation factors')
    parser.add_argument('--speed', type=float, default=1.0,
                      help='Speech speed (default: 1.0)')
    parser.add_argument('--lang', default='en-us',
                      help='Language code (default: en-us)')
    parser.add_argument('--output-dir', default='output',
                      help='Output directory for audio files')
    parser.add_argument('--db-path', default='voices.db',
                      help='Path to SQLite database')
    parser.add_argument('--insert', type=float, 
                      help='Insert interpolated voice at specified factor into database')
    parser.add_argument('--name', 
                      help='Name for the inserted voice (required with --insert)')
    parser.add_argument('--gender', choices=['M', 'F', 'X'],
                      help='Gender of the inserted voice (M/F/X)')
    parser.add_argument('--quality', type=int, choices=range(0, 101),
                      help='Quality rating of the inserted voice (0-100)')
    parser.add_argument('--notes',
                      help='Notes about the inserted voice (optional)')
    args = parser.parse_args()

    # Validate insert-related arguments
    if args.insert is not None:
        if not args.name:
            parser.error("--name is required when using --insert")
        if not args.gender:
            parser.error("--gender is required when using --insert")
        if args.quality is None:
            parser.error("--quality is required when using --insert")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Register the numpy array adapter
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    # Connect to database
    conn = sqlite3.connect(args.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    
    try:
        # Get voice group vectors
        print("\nProcessing source group...")
        source_style = get_voice_group_vector(conn, args.source_query)
        
        print("\nProcessing target group...")
        target_style = get_voice_group_vector(conn, args.target_query)
        
        # Initialize Kokoro
        kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
        
        # Handle single insertion if requested
        if args.insert is not None:
            print(f"\nGenerating voice for insertion at factor: {args.insert}")
            interpolated_style = interpolate_styles(source_style, target_style, args.insert)
            
            # Insert into database
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO voices 
            (name, gender, language, quality, style_vector, is_synthetic, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (
                args.name,
                args.gender,  # Now using provided gender
                args.lang,
                args.quality, # Now using provided quality
                interpolated_style,
                True, # Mark as synthetic
                args.notes
            ))
            conn.commit()
            print(f"Inserted synthetic voice '{args.name}' into database")
            
            # Generate sample for the inserted voice
            samples, sample_rate = kokoro.create(
                args.text,
                voice=interpolated_style,
                speed=args.speed,
                lang=args.lang
            )
            
            output_file = output_dir / f"{args.name}.wav"
            sf.write(output_file, samples, sample_rate)
            print(f"Created sample file: {output_file}")
            
        else:
            # Generate samples for each interpolation factor
            ranges = [float(x) for x in args.ranges.split(',')]
            
            for factor in ranges:
                print(f"\nGenerating sample for interpolation factor: {factor}")
                interpolated_style = interpolate_styles(source_style, target_style, factor)
                
                samples, sample_rate = kokoro.create(
                    args.text,
                    voice=interpolated_style,
                    speed=args.speed,
                    lang=args.lang
                )
                
                output_file = output_dir / f"interpolation_{factor:.2f}.wav"
                sf.write(output_file, samples, sample_rate)
                print(f"Created {output_file}")
                
    finally:
        conn.close()

if __name__ == "__main__":
    main()