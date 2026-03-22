"""
AI Soundscape Generator
=======================
Takes a natural-language scene description, inventories your sound library,
asks GPT-5.2 to design a rich layered soundscape, generates any missing
sounds via ElevenLabs Sound Effects API, then renders the final mix using
soundscape_composer.compose_soundscape().

Requirements:
    pip install openai elevenlabs pydub numpy
    # ffmpeg must be installed

Environment variables:
    OPENAI_API_KEY      — OpenAI API key
    ELEVENLABS_API_KEY  — ElevenLabs API key

Usage:
    python ai_soundscape.py --prompt "A peaceful Japanese garden at dawn with \
        a koi pond, wind chimes, and distant temple bells" \
        --duration 180 --library ./library --output output/japanese_garden.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from pathlib import Path

import openai
from elevenlabs import ElevenLabs

from soundscape_composer import (
    compose_soundscape,
    list_library,
    TrackLayer,
)


# ─── Configuration ──────────────────────────────────────────────────────────────

OPENAI_MODEL = "gpt-5.2"

ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"

# Where ElevenLabs-generated sounds get saved
GENERATED_DIR = "library/_generated"

# ─── System prompt for the planner ─────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert sound designer and foley artist. Your job is to design
immersive, cinematic soundscapes by composing multiple layered audio tracks.

You will be given:
1. A natural-language description of a scene.
2. The desired duration in seconds.
3. A catalogue of available sound files with their paths, names, durations,
   and folders.

Your output must be a single JSON object (no markdown, no commentary) with
exactly two top-level keys:

{
  "sounds_to_generate": [
    {
      "id": "gen_01",
      "description": "A short, clear description for an AI sound-effects
                       generator. Be specific about the sound, not the scene.
                       e.g. 'Gentle wooden wind chimes tinkling in a light
                       breeze' rather than 'Japanese garden ambience'.",
      "suggested_filename": "wind_chimes_gentle.mp3",
      "duration_seconds": 12.0
    }
  ],
  "layers": [
    {
      "file": "path/to/file.wav  OR  _generated/suggested_filename.mp3",
      "label": "Human-readable name",
      "volume_db": -6.0,
      "pan": 0.0,
      "loop": true,
      "loop_crossfade_ms": 150,
      "start_ms": 0,
      "end_ms": null,
      "fade_in_ms": 3000,
      "fade_out_ms": 3000,
      "fade_in_curve": "scurve",
      "fade_out_curve": "linear",
      "playback_rate": 1.0,
      "reverse": false,
      "low_pass_hz": null,
      "high_pass_hz": null,
      "occurrences": null,
      "random_occurrences": null,
      "volume_automation": null,
      "pan_automation": null
    }
  ]
}

Design principles:
- Build depth with 5–15 layers: a base ambience bed, mid-ground textures,
  foreground details, and sporadic one-shots for realism.
- Use pan (-1.0 to 1.0) to spread sounds across the stereo field.
  Don't put everything at center.
- Use volume_automation and pan_automation for movement and life.
  e.g. wind that sweeps, birds that move, traffic that swells.
- Use low_pass_hz to push sounds into the "distance" (400–1200 Hz).
- Use random_occurrences for natural sporadic sounds (birds, drips, cracks)
  with volume_var_db and pan_var for variation.
- Use occurrences for precisely timed dramatic moments.
- Loop ambient beds. Don't loop one-shots.
- Fade everything in/out — no hard cuts. Use scurve or logarithmic fades
  for natural-sounding transitions.
- Stagger start_ms so layers don't all begin at once — build the scene.
- Keep the master peak around -3 to -6 dB. Set base layers around -4 to -8 dB,
  details at -10 to -18 dB, and one-shots at -12 to -20 dB.
- If a sound in the library is close but not ideal, use it and adjust with
  playback_rate, low_pass_hz, high_pass_hz, or volume to reshape it.
- Only request generation for sounds that truly don't exist in the library
  and are essential for the scene. Prefer reusing library sounds creatively.
- For generated sounds, write descriptions that are specific and physical:
  describe the actual sound, not the mood. ElevenLabs needs concrete audio
  descriptions like "Heavy rain on a tin roof" not "Melancholic atmosphere".
- Keep generated sound durations practical: 5–15s for loops, 2–8s for one-shots.

For files from the library, use their exact path as listed in the catalogue.
For generated files, use: "_generated/<suggested_filename>".
The system will replace _generated/ with the actual path after generation.
""")


# ─── ElevenLabs sound generation ────────────────────────────────────────────────


def generate_sound_effect(
    client: ElevenLabs,
    description: str,
    filename: str,
    duration_seconds: float,
    output_dir: str = GENERATED_DIR,
) -> str:
    """
    Generate a sound effect using ElevenLabs and save it to disk.

    Returns the path to the saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # Skip if already generated (cache)
    if out_path.exists():
        print(f"  [cache] {filename} already exists, skipping generation")
        return str(out_path)

    print(f"  [elevenlabs] Generating: {description!r} → {filename} ({duration_seconds}s)")

    result = client.text_to_sound_effects.convert(
        text=description,
        duration_seconds=min(duration_seconds, 22.0),  # ElevenLabs max is 22s
        prompt_influence=0.5,
    )

    # result is a generator of bytes chunks
    audio_bytes = b"".join(chunk for chunk in result)

    out_path.write_bytes(audio_bytes)
    print(f"  [elevenlabs] Saved: {out_path} ({len(audio_bytes)} bytes)")

    return str(out_path)


# ─── GPT-5.2 planner ───────────────────────────────────────────────────────────


def plan_soundscape(
    scene_description: str,
    duration_seconds: float,
    library_catalogue: list[dict],
    openai_client: openai.OpenAI,
) -> dict:
    """
    Ask GPT-5.2 to design a soundscape composition plan.

    Returns the parsed JSON plan with 'sounds_to_generate' and 'layers'.
    """
    catalogue_text = json.dumps(library_catalogue, indent=2)

    user_message = textwrap.dedent(f"""\
    Scene description: {scene_description}

    Desired duration: {duration_seconds} seconds

    Available sound library:
    {catalogue_text}
    """)

    print(f"[gpt-5.2] Planning soundscape for: {scene_description!r}")
    print(f"[gpt-5.2] Library has {len(library_catalogue)} sounds available")

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    raw = response.choices[0].message.content
    plan = json.loads(raw)

    n_gen = len(plan.get("sounds_to_generate", []))
    n_layers = len(plan.get("layers", []))
    print(f"[gpt-5.2] Plan: {n_layers} layers, {n_gen} sounds to generate")

    return plan


# ─── Orchestrator ───────────────────────────────────────────────────────────────


def generate_soundscape(
    scene_description: str,
    duration_seconds: float = 120,
    library_dir: str = "library",
    output_path: str = "output/soundscape.wav",
    output_format: str = "wav",
    sample_rate: int = 44100,
    channels: int = 2,
    normalize: bool = True,
    master_volume_db: float = -1.0,
    master_fade_in_ms: int = 3000,
    master_fade_out_ms: int = 5000,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """
    End-to-end: describe a scene → get a rendered soundscape file.

    Parameters
    ----------
    scene_description : str
        Natural language description of the desired soundscape.
        e.g. "A bustling Tokyo street at night with rain, neon hum,
              distant sirens, and a street musician playing saxophone"

    duration_seconds : float
        Length of the output file in seconds.

    library_dir : str
        Path to the root sound library folder.

    output_path : str
        Where to save the final rendered file.

    output_format : str
        "wav", "mp3", "ogg", or "flac".

    sample_rate : int
        Output sample rate (default 44100).

    channels : int
        1 = mono, 2 = stereo (default 2).

    normalize : bool
        Normalize the final mix to prevent clipping.

    master_volume_db : float
        Master gain for the final mix.

    master_fade_in_ms : int
        Fade-in on the master output.

    master_fade_out_ms : int
        Fade-out on the master output.

    openai_api_key : str | None
        OpenAI API key. Falls back to OPENAI_API_KEY env var.

    elevenlabs_api_key : str | None
        ElevenLabs API key. Falls back to ELEVENLABS_API_KEY env var.

    Returns
    -------
    dict with keys:
        - path: output file path
        - duration_ms: actual duration
        - layers_processed: number of layers rendered
        - peak_db: peak level
        - clipped: whether clipping occurred
        - plan: the raw GPT-5.2 plan (for inspection/debugging)
        - generated_sounds: list of sounds that were created via ElevenLabs
    """

    # ── Resolve API keys ────────────────────────────────────────────────
    oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    el_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")

    if not oai_key:
        raise ValueError(
            "OpenAI API key required. Pass openai_api_key= or set OPENAI_API_KEY."
        )

    oai_client = openai.OpenAI(api_key=oai_key)

    # ── Scan the library ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AI Soundscape Generator")
    print(f"{'='*60}")
    print(f"  Scene:    {scene_description}")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Library:  {library_dir}")
    print(f"  Output:   {output_path}")
    print(f"{'='*60}\n")

    catalogue = list_library(library_dir)
    print(f"[library] Found {len(catalogue)} audio files\n")

    # ── Plan with GPT-5.2 ──────────────────────────────────────────────
    plan = plan_soundscape(
        scene_description=scene_description,
        duration_seconds=duration_seconds,
        library_catalogue=catalogue,
        openai_client=oai_client,
    )

    # ── Generate missing sounds via ElevenLabs ──────────────────────────
    generated_sounds = []
    sounds_to_gen = plan.get("sounds_to_generate", [])

    if sounds_to_gen:
        if not el_key:
            print("[warning] No ElevenLabs API key — skipping sound generation.")
            print("          Set ELEVENLABS_API_KEY or pass elevenlabs_api_key=.")
            print("          Layers using generated sounds will be silent/skipped.\n")
        else:
            el_client = ElevenLabs(api_key=el_key)
            gen_dir = str(Path(library_dir) / "_generated")

            print(f"\n[generate] {len(sounds_to_gen)} sounds to create:\n")

            for sound in sounds_to_gen:
                try:
                    path = generate_sound_effect(
                        client=el_client,
                        description=sound["description"],
                        filename=sound["suggested_filename"],
                        duration_seconds=sound.get("duration_seconds", 10.0),
                        output_dir=gen_dir,
                    )
                    generated_sounds.append({
                        "id": sound.get("id", ""),
                        "description": sound["description"],
                        "path": path,
                    })
                except Exception as e:
                    print(f"  [error] Failed to generate {sound['suggested_filename']}: {e}")

            print()

    # ── Resolve file paths in the plan ──────────────────────────────────
    gen_dir_prefix = "_generated/"
    layers = []

    for layer_data in plan.get("layers", []):
        file_path = layer_data.get("file", "")

        # Resolve _generated/ prefix to actual library path
        if file_path.startswith(gen_dir_prefix):
            file_path = str(
                Path(library_dir) / "_generated" / file_path[len(gen_dir_prefix):]
            )
        elif not Path(file_path).is_absolute():
            # Paths from catalogue are already relative to library_dir
            # but double-check they exist
            if not Path(file_path).exists():
                candidate = Path(library_dir) / file_path
                if candidate.exists():
                    file_path = str(candidate)

        layer_data["file"] = file_path

        # Clean up null values to let TrackLayer defaults apply
        cleaned = {}
        for k, v in layer_data.items():
            if v is not None:
                cleaned[k] = v
            elif k == "end_ms":
                # Explicitly keep None for end_ms (means "to end of soundscape")
                cleaned[k] = None

        layers.append(cleaned)

    # ── Render ──────────────────────────────────────────────────────────
    print(f"\n[render] Composing {len(layers)} layers into {duration_seconds}s soundscape...\n")

    result = compose_soundscape(
        output_path=output_path,
        duration_seconds=duration_seconds,
        layers=layers,
        master_volume_db=master_volume_db,
        fade_in_ms=master_fade_in_ms,
        fade_out_ms=master_fade_out_ms,
        sample_rate=sample_rate,
        channels=channels,
        output_format=output_format,
        normalize=normalize,
        normalize_headroom_db=-1.0,
    )

    result["plan"] = plan
    result["generated_sounds"] = generated_sounds

    print(f"[done] Exported: {result['path']}")
    print(f"       Duration: {result['duration_ms'] / 1000:.1f}s")
    print(f"       Layers:   {result['layers_processed']}")
    print(f"       Peak:     {result['peak_db']} dB")
    print(f"       Clipped:  {result['clipped']}")

    if generated_sounds:
        print(f"       Generated: {len(generated_sounds)} new sounds via ElevenLabs")

    return result


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Soundscape Generator — describe a scene, get an audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          %(prog)s --prompt "A rainy café in Paris with soft jazz, espresso machine,
                             quiet chatter, and rain on windows" --duration 300

          %(prog)s --prompt "Deep space: reactor hum, distant radio chatter,
                             occasional metallic creaks, warning beeps"
                   --duration 600 --output output/space_station.mp3 --format mp3

          %(prog)s --prompt "Tropical beach at sunset with waves, seagulls,
                             steel drums in the distance, and a bonfire"
                   --duration 180 --library ./my_sounds
        """),
    )

    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Natural language description of the soundscape.",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=120,
        help="Duration in seconds (default: 120).",
    )
    parser.add_argument(
        "--library", "-l",
        default="library",
        help="Path to the sound library folder (default: ./library).",
    )
    parser.add_argument(
        "--output", "-o",
        default="output/soundscape.wav",
        help="Output file path (default: output/soundscape.wav).",
    )
    parser.add_argument(
        "--format", "-f",
        default="wav",
        choices=["wav", "mp3", "ogg", "flac"],
        help="Output format (default: wav).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100).",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Output mono instead of stereo.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalization of the final mix.",
    )
    parser.add_argument(
        "--master-volume",
        type=float,
        default=-1.0,
        help="Master volume in dB (default: -1.0).",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--elevenlabs-key",
        default=None,
        help="ElevenLabs API key (or set ELEVENLABS_API_KEY env var).",
    )
    parser.add_argument(
        "--save-plan",
        default=None,
        help="Save the GPT-5.2 composition plan to a JSON file for inspection.",
    )

    args = parser.parse_args()

    result = generate_soundscape(
        scene_description=args.prompt,
        duration_seconds=args.duration,
        library_dir=args.library,
        output_path=args.output,
        output_format=args.format,
        sample_rate=args.sample_rate,
        channels=1 if args.mono else 2,
        normalize=not args.no_normalize,
        master_volume_db=args.master_volume,
        openai_api_key=args.openai_key,
        elevenlabs_api_key=args.elevenlabs_key,
    )

    if args.save_plan and result.get("plan"):
        plan_path = Path(args.save_plan)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(result["plan"], indent=2))
        print(f"\n[plan] Saved composition plan to: {plan_path}")


if __name__ == "__main__":
    main()
