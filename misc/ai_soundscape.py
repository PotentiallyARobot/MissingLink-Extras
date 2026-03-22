"""
AI Soundscape Generator + Refinement
=====================================
Two core functions:

    generate_soundscape(...)  → creates audio + writes a .score.json file
    refine_soundscape(...)    → reads the score, sends it + your instruction
                                to GPT-5.2, generates new sounds as needed,
                                re-renders, writes an updated .score.json

The .score.json is the persistent textual representation of the composition.
It contains the full layer spec, generated sounds manifest, scene description,
and duration — everything the LLM needs to understand and modify the piece.

Requirements:
    pip install openai elevenlabs pydub numpy
    # ffmpeg must be installed

Environment variables:
    OPENAI_API_KEY      — OpenAI API key
    ELEVENLABS_API_KEY  — ElevenLabs API key
"""

from __future__ import annotations

import json
import os
import textwrap
import warnings
from pathlib import Path
from datetime import datetime

import openai
from elevenlabs import ElevenLabs

from soundscape_composer import compose_soundscape, list_library


# ─── Config ─────────────────────────────────────────────────────────────────────

OPENAI_MODEL = "gpt-5.2"


# ─── Score file I/O ─────────────────────────────────────────────────────────────

def _score_path(audio_path: str) -> str:
    """Derive the .score.json path from the audio output path."""
    p = Path(audio_path)
    return str(p.with_suffix(".score.json"))


def write_score(
    audio_path: str,
    plan: dict,
    scene_description: str,
    duration_seconds: float,
    library_dir: str,
    revision: int = 0,
    history: list[str] | None = None,
) -> str:
    """
    Write the full score file next to the audio file.
    Returns the score file path.

    The score contains everything needed to understand, reproduce,
    or further refine the soundscape:
      - scene_description: the original prompt
      - duration_seconds
      - revision: how many refinements have been applied
      - history: list of all instructions (original + refinements)
      - library_dir: where sounds live
      - sounds_generated: manifest of ElevenLabs-created sounds
      - layers: the full composition spec (every parameter)
      - human_readable_score: a plain-text breakdown for quick reading
    """
    score_file = _score_path(audio_path)

    # Build human-readable text version of the score
    readable_lines = []
    readable_lines.append(f"SOUNDSCAPE: {scene_description}")
    readable_lines.append(f"Duration: {duration_seconds}s | Revision: {revision}")
    readable_lines.append(f"Layers: {len(plan.get('layers', []))}")
    readable_lines.append("")

    for i, L in enumerate(plan.get("layers", []), 1):
        readable_lines.append(f"[{i}] {L.get('label', 'Untitled')}")
        readable_lines.append(f"    file: {L.get('file', '?')}")
        readable_lines.append(f"    vol: {L.get('volume_db', 0)}dB  pan: {L.get('pan', 0)}  loop: {L.get('loop', False)}")
        readable_lines.append(f"    window: {L.get('start_ms', 0)}ms -> {L.get('end_ms', 'END')}")
        readable_lines.append(f"    fade_in: {L.get('fade_in_ms', 0)}ms ({L.get('fade_in_curve', 'linear')})  fade_out: {L.get('fade_out_ms', 0)}ms ({L.get('fade_out_curve', 'linear')})")

        if L.get("low_pass_hz") or L.get("high_pass_hz"):
            readable_lines.append(f"    filters: LP={L.get('low_pass_hz', '-')}Hz  HP={L.get('high_pass_hz', '-')}Hz")
        if L.get("playback_rate", 1.0) != 1.0:
            readable_lines.append(f"    rate: {L['playback_rate']}x")
        if L.get("reverse"):
            readable_lines.append(f"    REVERSED")

        if L.get("volume_automation"):
            pts = " -> ".join(
                f"{p.get('time_ms', 0)}ms:{p.get('value', p.get('value_db', 0))}dB"
                for p in L["volume_automation"]
            )
            readable_lines.append(f"    vol_auto: {pts}")
        if L.get("pan_automation"):
            pts = " -> ".join(
                f"{p.get('time_ms', 0)}ms:{p.get('value', p.get('pan', 0))}"
                for p in L["pan_automation"]
            )
            readable_lines.append(f"    pan_auto: {pts}")
        if L.get("occurrences"):
            readable_lines.append(f"    occurrences: {len(L['occurrences'])} hits")
            for j, occ in enumerate(L["occurrences"][:6]):
                t = occ.get("start_ms", occ.get("time_ms", "?"))
                readable_lines.append(f"      @{t}ms  pan={occ.get('pan', '-')}  vol={occ.get('volume_db', occ.get('gain_db', '-'))}dB")
            if len(L["occurrences"]) > 6:
                readable_lines.append(f"      ... +{len(L['occurrences'])-6} more")
        if L.get("random_occurrences"):
            ro = L["random_occurrences"]
            if isinstance(ro, dict):
                readable_lines.append(f"    random: count={ro.get('count','?')} gap>={ro.get('min_gap_ms','?')}ms vol+-{ro.get('volume_var_db','?')}dB pan+-{ro.get('pan_var','?')}")
        readable_lines.append("")

    gens = plan.get("sounds_to_generate", [])
    if gens:
        readable_lines.append("GENERATED SOUNDS:")
        for g in gens:
            readable_lines.append(f"  * {g.get('suggested_filename', '?')}: {g.get('description', '?')} ({g.get('duration_seconds', '?')}s)")

    score_data = {
        "scene_description": scene_description,
        "duration_seconds": duration_seconds,
        "revision": revision,
        "history": history or [scene_description],
        "library_dir": library_dir,
        "sounds_generated": plan.get("sounds_to_generate", []),
        "layers": plan.get("layers", []),
        "human_readable_score": "\n".join(readable_lines),
    }

    Path(score_file).parent.mkdir(parents=True, exist_ok=True)
    Path(score_file).write_text(json.dumps(score_data, indent=2))
    print(f"[score] Written: {score_file}")
    return score_file


def read_score(score_path: str) -> dict:
    """Load a score file."""
    return json.loads(Path(score_path).read_text())


# ─── ElevenLabs generation ──────────────────────────────────────────────────────

def _generate_sound(
    client: ElevenLabs,
    description: str,
    filename: str,
    duration_seconds: float,
    output_dir: str,
) -> str:
    """Generate a single sound via ElevenLabs. Caches by filename."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if out_path.exists():
        print(f"  [cache] {filename}")
        return str(out_path)

    print(f"  [elevenlabs] {description!r} -> {filename} ({duration_seconds}s)")
    result = client.text_to_sound_effects.convert(
        text=description,
        duration_seconds=min(duration_seconds, 22.0),
        prompt_influence=0.5,
    )
    audio_bytes = b"".join(chunk for chunk in result)
    out_path.write_bytes(audio_bytes)
    print(f"  [elevenlabs] Saved ({len(audio_bytes)} bytes)")
    return str(out_path)


def _generate_all(
    sounds: list[dict],
    library_dir: str,
    el_client: ElevenLabs | None,
) -> list[dict]:
    """Generate all requested sounds. Returns manifest of what was created."""
    if not sounds:
        return []

    if not el_client:
        print(f"[warning] {len(sounds)} sounds need generation but no ElevenLabs client.")
        return []

    gen_dir = str(Path(library_dir) / "_generated")
    generated = []
    print(f"\n[generate] {len(sounds)} sounds:\n")

    for s in sounds:
        try:
            path = _generate_sound(
                client=el_client,
                description=s["description"],
                filename=s["suggested_filename"],
                duration_seconds=s.get("duration_seconds", 10.0),
                output_dir=gen_dir,
            )
            generated.append({"id": s.get("id", ""), "description": s["description"], "path": path})
        except Exception as e:
            print(f"  [error] {s.get('suggested_filename', '?')}: {e}")

    return generated


# ─── Path resolution ────────────────────────────────────────────────────────────

def _resolve_paths(layers: list[dict], library_dir: str) -> list[dict]:
    """Resolve _generated/ prefixes and relative paths."""
    resolved = []
    for layer in layers:
        layer = dict(layer)
        fp = layer.get("file", "")

        if fp.startswith("_generated/"):
            fp = str(Path(library_dir) / fp)
        elif not Path(fp).is_absolute() and not Path(fp).exists():
            candidate = Path(library_dir) / fp
            if candidate.exists():
                fp = str(candidate)

        layer["file"] = fp

        # Clean nulls (keep end_ms=None explicitly)
        cleaned = {}
        for k, v in layer.items():
            if v is not None:
                cleaned[k] = v
            elif k == "end_ms":
                cleaned[k] = None
        resolved.append(cleaned)
    return resolved


# ─── Render helper ───────────────────────────────────────────────────────────────

def _render(
    layers: list[dict],
    output_path: str,
    duration_seconds: float,
    output_format: str = "wav",
    sample_rate: int = 44100,
    channels: int = 2,
    master_volume_db: float = -1.0,
    master_fade_in_ms: int = 3000,
    master_fade_out_ms: int = 5000,
) -> dict:
    """Render layers to an audio file."""
    print(f"\n[render] {len(layers)} layers -> {duration_seconds}s ...\n")
    return compose_soundscape(
        output_path=output_path,
        duration_seconds=duration_seconds,
        layers=layers,
        master_volume_db=master_volume_db,
        fade_in_ms=master_fade_in_ms,
        fade_out_ms=master_fade_out_ms,
        sample_rate=sample_rate,
        channels=channels,
        output_format=output_format,
        normalize=True,
        normalize_headroom_db=-1.0,
    )


# ─── System prompts ─────────────────────────────────────────────────────────────

_SCHEMA_RULES = """\
OUTPUT FORMAT: A single JSON object with two keys: "sounds_to_generate" and "layers".
No markdown, no commentary, no backticks. Raw JSON only.

SCHEMA RULES (follow exactly or the renderer will crash):
- volume_automation points: {"time_ms": int, "value": float}  (NOT "value_db")
- pan_automation points: {"time_ms": int, "value": float}  (NOT "pan")
- occurrences: [{"start_ms": int, "volume_db": float, "pan": float, ...}]
  Valid occurrence keys ONLY: start_ms, volume_db, pan, fade_in_ms, fade_out_ms,
  playback_rate, trim_start_ms, trim_end_ms, reverse. NO other keys like
  "gain_db", "time_ms", "pitch_var_semitones" etc.
- random_occurrences MUST be a DICT with keys: count, min_gap_ms,
  volume_var_db, pan_var, rate_var.  It must NEVER be a list.
- Layer valid keys ONLY: file, label, volume_db, pan, loop, loop_crossfade_ms,
  start_ms, end_ms, fade_in_ms, fade_out_ms, fade_in_curve, fade_out_curve,
  playback_rate, reverse, low_pass_hz, high_pass_hz, occurrences,
  random_occurrences, volume_automation, pan_automation.
  Do NOT invent keys like "min_interval_ms", "max_interval_ms",
  "pitch_var_semitones" etc.
- For generated sounds, set file to: "_generated/<filename>"
- For existing library sounds, use their exact path from the catalogue.
- sounds_to_generate entries: {"id": str, "description": str,
  "suggested_filename": str, "duration_seconds": float}.
  Description must be concrete and physical (the actual sound, not mood).
  Keep durations practical: 5-15s for loops, 2-8s for one-shots.
- fade_in_curve / fade_out_curve valid values: "linear", "logarithmic",
  "exponential", "scurve"."""

CREATE_SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an expert sound designer. Design immersive, cinematic soundscapes
by composing multiple layered audio tracks.

You will receive a scene description, duration, and a catalogue of
available sound files.

{_SCHEMA_RULES}

MIXING PRINCIPLES:
- Build depth with 5-15 layers: base ambient bed, mid-ground textures,
  foreground details, sporadic one-shots.
- Spread sounds across the stereo field with pan (-1.0 to 1.0).
- Use volume_automation and pan_automation for movement and life.
- Use low_pass_hz (400-1200) to push sounds into the "distance".
- Use random_occurrences for natural sporadic sounds with variation.
- Loop ambient beds. Don't loop one-shots.
- Fade everything. No hard cuts. Use scurve or logarithmic curves.
- Stagger start_ms so layers build the scene gradually.
- Base beds: -4 to -8 dB. Details: -10 to -18 dB. One-shots: -12 to -20 dB.
- Prefer reusing library sounds creatively (rate, filters, reverse) over
  generating new ones. Only generate what truly does not exist.
""")

REFINE_SYSTEM_PROMPT = textwrap.dedent(f"""\
You are an expert sound designer refining an existing soundscape composition.

You will receive:
1. The current composition score (full layer-by-layer breakdown).
2. The user's corrective instruction.
3. The available sound library catalogue.
4. The total duration.

{_SCHEMA_RULES}

REFINEMENT RULES:
- PRESERVE layers the user did not mention. Do not drop untouched layers.
- When told "make X louder/quieter" -> adjust volume_db or volume_automation.
- When told "add Y" -> design a new layer. Generate the sound if not in library.
- When told "remove Y" -> drop that layer entirely.
- When told "X sounds bad/wrong" -> either replace the sound file
  (generate if needed) or heavily adjust parameters (filters, volume, timing).
- When told qualitative things like "more dramatic", "calmer", "build tension"
  -> interpret via automation curves, volume shifts, new layers, timing changes.
- When told "move X earlier/later" -> shift start_ms / end_ms / occurrences.
- You can split one layer into two, merge layers, or restructure freely.
- Always output the COMPLETE layer list including unchanged layers.
""")


# ─── GPT calls ──────────────────────────────────────────────────────────────────

def _call_gpt(
    system_prompt: str,
    user_message: str,
    oai_client: openai.OpenAI,
    temperature: float = 0.7,
) -> dict:
    """Call GPT-5.2 and parse JSON response."""
    response = oai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return json.loads(response.choices[0].message.content)


# ─── generate_soundscape ────────────────────────────────────────────────────────

def generate_soundscape(
    scene_description: str,
    duration_seconds: float = 120,
    library_dir: str = "library",
    output_path: str = "output/soundscape.wav",
    output_format: str = "wav",
    sample_rate: int = 44100,
    channels: int = 2,
    master_volume_db: float = -1.0,
    master_fade_in_ms: int = 3000,
    master_fade_out_ms: int = 5000,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """
    Generate a soundscape from a natural-language scene description.

    Produces:
      1. The audio file at output_path
      2. A .score.json next to it with the full composition spec

    The .score.json is what refine_soundscape() reads to understand
    and modify the composition.

    Returns dict with: path, duration_ms, layers_processed, peak_db,
    clipped, plan, generated_sounds, score_file.
    """
    oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    el_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not oai_key:
        raise ValueError("OpenAI API key required.")

    oai_client = openai.OpenAI(api_key=oai_key)
    el_client = ElevenLabs(api_key=el_key) if el_key else None

    # Scan library
    catalogue = list_library(library_dir)
    print(f"[library] {len(catalogue)} audio files found")

    # Plan
    print(f"[gpt-5.2] Planning: {scene_description!r}")
    user_msg = (
        f"Scene: {scene_description}\n"
        f"Duration: {duration_seconds} seconds\n\n"
        f"Available library:\n{json.dumps(catalogue, indent=2)}"
    )
    plan = _call_gpt(CREATE_SYSTEM_PROMPT, user_msg, oai_client)

    n_gen = len(plan.get("sounds_to_generate", []))
    n_layers = len(plan.get("layers", []))
    print(f"[gpt-5.2] Plan: {n_layers} layers, {n_gen} sounds to generate")

    # Generate missing sounds
    generated = _generate_all(plan.get("sounds_to_generate", []), library_dir, el_client)

    # Resolve paths and render
    layers = _resolve_paths(plan.get("layers", []), library_dir)
    render_result = _render(
        layers, output_path, duration_seconds, output_format,
        sample_rate, channels, master_volume_db,
        master_fade_in_ms, master_fade_out_ms,
    )

    # Write score
    score_file = write_score(
        audio_path=output_path,
        plan=plan,
        scene_description=scene_description,
        duration_seconds=duration_seconds,
        library_dir=library_dir,
        revision=0,
        history=[scene_description],
    )

    render_result["plan"] = plan
    render_result["generated_sounds"] = generated
    render_result["score_file"] = score_file

    print(f"\n[done] Audio: {render_result['path']}")
    print(f"       Score: {score_file}")
    print(f"       Layers: {render_result['layers_processed']}  Peak: {render_result['peak_db']}dB")

    return render_result


# ─── refine_soundscape ──────────────────────────────────────────────────────────

def refine_soundscape(
    instruction: str,
    score_path: str | None = None,
    audio_path: str | None = None,
    output_path: str | None = None,
    output_format: str = "wav",
    sample_rate: int = 44100,
    channels: int = 2,
    master_volume_db: float = -1.0,
    master_fade_in_ms: int = 3000,
    master_fade_out_ms: int = 5000,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """
    Refine an existing soundscape by sending a corrective instruction.

    Reads the .score.json, sends the full human-readable score + raw layer
    JSON + your instruction + the current library catalogue to GPT-5.2.
    GPT returns a revised plan. Any new sounds are generated via ElevenLabs.
    The composition is re-rendered and a new .score.json is written.

    Parameters
    ----------
    instruction : str
        Natural-language instruction describing what to change.
        Examples:
          "Remove the cockpit beeps, they sound unrealistic"
          "Make the wind much more aggressive in the first 60 seconds"
          "Add a deep metallic groaning sound throughout the descent"
          "The landing needs more impact, add a shockwave boom at touchdown"
          "Everything is too busy, simplify to just wind and thrusters"
          "Pan the hull creaks to the left and make them quieter"
          "Add a radio static crackle that fades in around 90 seconds"

    score_path : str | None
        Path to the .score.json file. If None, derived from audio_path.

    audio_path : str | None
        Path to the previous audio file. Used to find the score if
        score_path is not given.

    output_path : str | None
        Where to write the new audio. If None, auto-generates a
        versioned filename next to the score.

    Returns dict with: path, duration_ms, layers_processed, peak_db,
    clipped, plan, generated_sounds, score_file.
    """
    # Resolve score path
    if score_path is None and audio_path is not None:
        score_path = _score_path(audio_path)
    if score_path is None:
        raise ValueError("Provide either score_path or audio_path.")
    if not Path(score_path).exists():
        raise FileNotFoundError(f"Score not found: {score_path}")

    oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    el_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY")
    if not oai_key:
        raise ValueError("OpenAI API key required.")

    oai_client = openai.OpenAI(api_key=oai_key)
    el_client = ElevenLabs(api_key=el_key) if el_key else None

    # Load current score
    score = read_score(score_path)
    scene = score["scene_description"]
    duration = score["duration_seconds"]
    library_dir = score["library_dir"]
    revision = score.get("revision", 0) + 1
    history = score.get("history", [scene])

    print(f"[refine] Revision {revision}")
    print(f"[refine] Instruction: {instruction!r}")
    print(f"[refine] Current: {len(score.get('layers', []))} layers")

    # Scan library (includes previously generated sounds)
    catalogue = list_library(library_dir)

    # Build user message with full context
    user_msg = (
        f"CURRENT COMPOSITION SCORE:\n"
        f"{score['human_readable_score']}\n\n"
        f"FULL LAYER DATA (JSON):\n"
        f"{json.dumps(score['layers'], indent=2)}\n\n"
        f"DURATION: {duration} seconds\n\n"
        f"AVAILABLE LIBRARY ({len(catalogue)} files):\n"
        f"{json.dumps(catalogue, indent=2)}\n\n"
        f"INSTRUCTION:\n{instruction}"
    )

    # Get revised plan
    print(f"[gpt-5.2] Refining...")
    new_plan = _call_gpt(REFINE_SYSTEM_PROMPT, user_msg, oai_client, temperature=0.6)

    n_gen = len(new_plan.get("sounds_to_generate", []))
    n_layers = len(new_plan.get("layers", []))
    print(f"[gpt-5.2] Revised: {n_layers} layers, {n_gen} new sounds")

    # Generate new sounds
    generated = _generate_all(new_plan.get("sounds_to_generate", []), library_dir, el_client)

    # Resolve paths and render
    layers = _resolve_paths(new_plan.get("layers", []), library_dir)

    if output_path is None:
        base = Path(score_path).stem.replace(".score", "")
        ts = datetime.now().strftime("%H%M%S")
        output_path = str(Path(score_path).parent / f"{base}_v{revision}_{ts}.wav")

    render_result = _render(
        layers, output_path, duration, output_format,
        sample_rate, channels, master_volume_db,
        master_fade_in_ms, master_fade_out_ms,
    )

    # Write updated score
    history = history + [instruction]
    score_file = write_score(
        audio_path=output_path,
        plan=new_plan,
        scene_description=scene,
        duration_seconds=duration,
        library_dir=library_dir,
        revision=revision,
        history=history,
    )

    render_result["plan"] = new_plan
    render_result["generated_sounds"] = generated
    render_result["score_file"] = score_file

    print(f"\n[done] Audio: {render_result['path']}")
    print(f"       Score: {score_file}")
    print(f"       Layers: {render_result['layers_processed']}  Peak: {render_result['peak_db']}dB")
    print(f"       Revision: {revision}  History: {len(history)} instructions")

    return render_result
