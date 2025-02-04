# KokoVoiceLab

An application for experimenting with Kokoro voice models, allowing voice interpolation and managing voices in a database for later use. The application uses sqlite queries to select two different groups of voices from the available kokoro voice dataset. It then uses the differences of those groups to create interpolation between **and** beyond them. The usage example shows grouping of quality American and British voices. When the code is run it produces audio from extreme American (-2) to extreme British (2) which goes beyond the accent trait of the available models.

This setup allows for the synthetic generation of voices that are not present in the original dataset. You can then insert those voices into the database and mix them to make even more diverse and unique voices.

## Installation
I highly encourage the use of uv for dependency management.

1. Ensure Python 3.12+ is installed
2. Install uv (https://github.com/astral-sh/uv):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Download required model files:

```bash
uv run scripts/fetch_models.py
```

4. Initialize the voice database:

```bash
uv run scripts/create_voice_db.py
```

This will create a `voices.db` file containing voice metadata and style vectors.

## Usage Examples

### Voice Interpolation

Generate samples interpolating between high-quality female and male American English voices:

```bash
uv run kokovoicelab.py \
  --source-query "SELECT * FROM voices WHERE gender='F' AND language='American English' AND quality >= 70" \
  --target-query "SELECT * FROM voices WHERE gender='F' AND language='British English' AND quality >= 70" \
  --text "This is an example of how we can explore accents and other things. Neat!" \
  --ranges="-2,-1,0,1,2" \
  --output-dir "samples"
```

### Creating Custom Voices

Insert a new synthetic voice that's 70% between source and target:

```bash
uv run kokovoicelab.py \
  --source-query "SELECT * FROM voices WHERE name='af_heart'" \
  --target-query "SELECT * FROM voices WHERE name='am_fenrir'" \
  --text "This is a custom synthetic voice." \
  --insert 0.7 \
  --name "custom_voice_1" \
  --gender "X" \
  --quality 85 \
  --notes "70% interpolation between af_heart and am_fenrir" \
  --output-dir "custom_voices"
```

### Basic Voice Synthesis

Generate audio using a specific voice from the database (you can use your custom voices here):

```bash
uv run scripts/synthesize.py \
  --text "Hello, this is a test of the voice synthesis system." \
  --voice-name "af_heart" \
  --speed 1.0 \
  --output-dir "samples"
```

This will:
1. Load the specified voice from the database
2. Generate audio using the provided text
3. Save the result as a WAV file in the output directory

You can adjust the speech speed using the `--speed` parameter (default: 1.0).

### SQLite Schema

The database includes the following fields for each voice:
- `name`: Unique identifier
- `gender`: M/F/X
- `language`: Voice language
- `quality`: Rating from 0-100
- `training_duration`: Amount of training data used
- `style_vector`: Neural voice embedding
- `is_synthetic`: Boolean flag for generated voices
- `notes`: Optional description
- `created_at`: Timestamp

## Available Voices

https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md