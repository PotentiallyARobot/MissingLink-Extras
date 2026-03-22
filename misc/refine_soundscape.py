"""
Interactive Soundscape Refinement Loop (Colab)
===============================================
Paste this into a Colab cell after running the initial generate_soundscape() call.

Flow:
  1. Plays the current soundscape
  2. Shows a full textual description of every layer (the "score")
  3. Accepts your natural-language feedback
  4. Sends the current score + your feedback + library catalogue to GPT-5.2
  5. GPT returns a revised plan — new sounds are generated via ElevenLabs
  6. Re-renders and loops back to step 1
  7. Type "done" to exit

Requirements:
  - ai_soundscape.py and soundscape_composer.py on sys.path
  - OPENAI_API_KEY and ELEVENLABS_API_KEY set in env
  - `result` dict from a prior generate_soundscape() call
"""

import json
import os
import textwrap
import copy
from pathlib import Path
from datetime import datetime

import openai
from IPython.display import Audio, display, HTML, clear_output

from soundscape_composer import compose_soundscape, list_library
from ai_soundscape import generate_sound_effect

try:
    from elevenlabs import ElevenLabs
except ImportError:
    ElevenLabs = None


# ─── Score serializer ───────────────────────────────────────────────────────────

def plan_to_score(plan: dict, duration_seconds: float) -> str:
    """
    Convert a GPT composition plan into a detailed human+AI readable
    textual score that fully describes the soundscape.
    """
    lines = []
    lines.append(f"SOUNDSCAPE SCORE — {duration_seconds}s total")
    lines.append("=" * 70)

    layers = plan.get("layers", [])
    for i, L in enumerate(layers, 1):
        lines.append(f"\n── Layer {i}: {L.get('label', 'Untitled')} ──")
        lines.append(f"   File:          {L.get('file', '?')}")
        lines.append(f"   Volume:        {L.get('volume_db', 0)} dB")
        lines.append(f"   Pan:           {L.get('pan', 0)}")
        lines.append(f"   Loop:          {L.get('loop', False)}")
        if L.get("loop"):
            lines.append(f"   Loop XFade:    {L.get('loop_crossfade_ms', 0)} ms")
        lines.append(f"   Time window:   {L.get('start_ms', 0)} ms → {L.get('end_ms', 'end')}")
        lines.append(f"   Fade in:       {L.get('fade_in_ms', 0)} ms ({L.get('fade_in_curve', 'linear')})")
        lines.append(f"   Fade out:      {L.get('fade_out_ms', 0)} ms ({L.get('fade_out_curve', 'linear')})")
        lines.append(f"   Playback rate: {L.get('playback_rate', 1.0)}")
        lines.append(f"   Reverse:       {L.get('reverse', False)}")

        if L.get("low_pass_hz"):
            lines.append(f"   Low-pass:      {L['low_pass_hz']} Hz")
        if L.get("high_pass_hz"):
            lines.append(f"   High-pass:     {L['high_pass_hz']} Hz")

        if L.get("volume_automation"):
            pts = ", ".join(
                f"{p.get('time_ms', p.get('time', 0))}ms→{p.get('value', p.get('value_db', 0))}dB"
                for p in L["volume_automation"]
            )
            lines.append(f"   Vol automation: [{pts}]")

        if L.get("pan_automation"):
            pts = ", ".join(
                f"{p.get('time_ms', p.get('time', 0))}ms→{p.get('value', p.get('pan', 0))}"
                for p in L["pan_automation"]
            )
            lines.append(f"   Pan automation: [{pts}]")

        if L.get("occurrences"):
            lines.append(f"   Occurrences:   {len(L['occurrences'])} placements")
            for j, occ in enumerate(L["occurrences"]):
                t = occ.get("start_ms", occ.get("time_ms", "?"))
                lines.append(f"     #{j+1} at {t} ms  pan={occ.get('pan', '-')}  vol={occ.get('volume_db', occ.get('gain_db', '-'))} dB")

        if L.get("random_occurrences"):
            ro = L["random_occurrences"]
            if isinstance(ro, dict):
                lines.append(f"   Random:        {json.dumps(ro)}")
            elif isinstance(ro, list):
                lines.append(f"   Random (list): {len(ro)} placements")

    # Generated sounds catalogue
    gens = plan.get("sounds_to_generate", [])
    if gens:
        lines.append(f"\n{'='*70}")
        lines.append(f"GENERATED SOUNDS ({len(gens)}):")
        for g in gens:
            lines.append(f"  • {g.get('suggested_filename', '?')}: {g.get('description', '?')} ({g.get('duration_seconds', '?')}s)")

    return "\n".join(lines)


# ─── Refinement system prompt ───────────────────────────────────────────────────

REFINE_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert sound designer refining a soundscape composition.

You will receive:
1. The current SOUNDSCAPE SCORE — a textual description of every layer,
   its volume, panning, automation, timing, effects, and file.
2. The user's feedback / instructions for changes.
3. The available sound library catalogue (files already on disk).
4. The total duration in seconds.

Your job is to output a REVISED JSON plan that incorporates the user's
feedback. The JSON must have exactly two keys:

{
  "sounds_to_generate": [ ... ],
  "layers": [ ... ]
}

These follow the exact same schema as the original plan. Rules:

- PRESERVE layers the user didn't mention changing. Don't drop layers
  unless the user explicitly asks to remove them.
- When the user says "make X louder/quieter", adjust volume_db or
  volume_automation accordingly.
- When the user says "add Y", design a new layer. If the sound doesn't
  exist in the library, add it to sounds_to_generate with a concrete
  physical description of the sound (not mood words).
- When the user says "remove Y", drop that layer from the list.
- When the user says things like "more dramatic", "build tension",
  "calmer ending" — interpret creatively using volume automation,
  pan movement, filter changes, timing shifts, or new layers.
- You can reuse any file already on disk (from the library or previously
  generated in library/_generated/).
- For generated sounds, use "_generated/<filename>" as the file path.
- Keep the same mixing principles: base beds -4 to -8 dB, details -10
  to -18 dB, one-shots -12 to -20 dB. Use fades, automation, panning.
- Output ONLY the JSON object. No markdown, no commentary, no backticks.
- All volume_automation points must use the key "value" (not "value_db").
- All pan_automation points must use the key "value" (not "pan").
- random_occurrences must be a dict with keys: count, min_gap_ms,
  volume_var_db, pan_var, rate_var — NOT a list of occurrences.
- occurrences must use "start_ms" for timing (not "time_ms").
- Do not invent keys that don't exist in the TrackLayer schema.
  Valid keys: file, label, volume_db, pan, loop, loop_crossfade_ms,
  start_ms, end_ms, fade_in_ms, fade_out_ms, fade_in_curve,
  fade_out_curve, playback_rate, reverse, low_pass_hz, high_pass_hz,
  occurrences, random_occurrences, volume_automation, pan_automation.
- Valid occurrence keys: start_ms, volume_db, pan, fade_in_ms,
  fade_out_ms, playback_rate, trim_start_ms, trim_end_ms, reverse.
""")


# ─── Refinement call ────────────────────────────────────────────────────────────

def refine_plan(
    current_plan: dict,
    score_text: str,
    user_feedback: str,
    duration_seconds: float,
    library_catalogue: list[dict],
    openai_client: openai.OpenAI,
) -> dict:
    """Send the current score + feedback to GPT-5.2 and get a revised plan."""

    user_msg = textwrap.dedent(f"""\
    CURRENT SOUNDSCAPE SCORE:
    {score_text}

    DURATION: {duration_seconds} seconds

    AVAILABLE LIBRARY:
    {json.dumps(library_catalogue, indent=2)}

    USER FEEDBACK:
    {user_feedback}

    Please output the revised JSON plan.
    """)

    print(f"[gpt-5.2] Refining based on: {user_feedback!r}")

    response = openai_client.chat.completions.create(
        model="gpt-5.2",
        temperature=0.6,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": REFINE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    raw = response.choices[0].message.content
    plan = json.loads(raw)

    n_gen = len(plan.get("sounds_to_generate", []))
    n_layers = len(plan.get("layers", []))
    print(f"[gpt-5.2] Revised: {n_layers} layers, {n_gen} new sounds to generate")

    return plan


# ─── Path resolver ───────────────────────────────────────────────────────────────

def resolve_layer_paths(layers: list[dict], library_dir: str) -> list[dict]:
    """Resolve _generated/ prefixes and relative paths to absolute paths."""
    resolved = []
    for layer_data in layers:
        layer = dict(layer_data)
        fp = layer.get("file", "")

        if fp.startswith("_generated/"):
            fp = str(Path(library_dir) / fp)
        elif not Path(fp).is_absolute() and not Path(fp).exists():
            candidate = Path(library_dir) / fp
            if candidate.exists():
                fp = str(candidate)

        layer["file"] = fp

        # Strip nulls except end_ms
        cleaned = {}
        for k, v in layer.items():
            if v is not None:
                cleaned[k] = v
            elif k == "end_ms":
                cleaned[k] = None
        resolved.append(cleaned)

    return resolved


# ─── Main interactive loop ──────────────────────────────────────────────────────

def refinement_loop(
    result: dict,
    library_dir: str = "/content/library",
    output_dir: str = "/content/output",
    duration_seconds: float = 180,
    sample_rate: int = 44100,
    channels: int = 2,
):
    """
    Interactive refinement loop. Call this after generate_soundscape().

    Parameters
    ----------
    result : dict
        The return value from generate_soundscape(). Must contain 'plan' and 'path'.
    library_dir : str
        Path to the sound library.
    output_dir : str
        Where to save revised renders.
    duration_seconds : float
        Duration of the soundscape.
    """

    oai_key = os.environ.get("OPENAI_API_KEY")
    el_key = os.environ.get("ELEVENLABS_API_KEY")

    if not oai_key:
        raise ValueError("Set OPENAI_API_KEY environment variable.")

    oai_client = openai.OpenAI(api_key=oai_key)
    el_client = ElevenLabs(api_key=el_key) if el_key and ElevenLabs else None

    current_plan = copy.deepcopy(result["plan"])
    current_path = result["path"]
    revision = 0

    print("=" * 60)
    print("  INTERACTIVE SOUNDSCAPE REFINEMENT")
    print("  Type your changes, or 'done' to finish.")
    print("  Type 'score' to see the current composition.")
    print("  Type 'replay' to hear the current version again.")
    print("=" * 60)

    # Initial playback
    print(f"\n▶ Playing current soundscape ({duration_seconds}s):\n")
    display(Audio(current_path))

    score_text = plan_to_score(current_plan, duration_seconds)
    print(f"\n{score_text}\n")

    while True:
        print("-" * 60)
        feedback = input("\n🎧 Your feedback (or 'done'): ").strip()

        if not feedback:
            continue

        if feedback.lower() == "done":
            print(f"\n✅ Final soundscape: {current_path}")
            print(f"   Revisions made: {revision}")
            display(Audio(current_path))
            break

        if feedback.lower() == "score":
            score_text = plan_to_score(current_plan, duration_seconds)
            print(f"\n{score_text}\n")
            continue

        if feedback.lower() == "replay":
            display(Audio(current_path))
            continue

        # ── Refine with GPT-5.2 ──────────────────────────────────────────
        catalogue = list_library(library_dir)
        score_text = plan_to_score(current_plan, duration_seconds)

        try:
            new_plan = refine_plan(
                current_plan=current_plan,
                score_text=score_text,
                user_feedback=feedback,
                duration_seconds=duration_seconds,
                library_catalogue=catalogue,
                openai_client=oai_client,
            )
        except Exception as e:
            print(f"[error] GPT refinement failed: {e}")
            continue

        # ── Generate any new sounds ──────────────────────────────────────
        sounds_to_gen = new_plan.get("sounds_to_generate", [])
        if sounds_to_gen and el_client:
            gen_dir = str(Path(library_dir) / "_generated")
            print(f"\n[generate] {len(sounds_to_gen)} new sounds:\n")
            for sound in sounds_to_gen:
                try:
                    generate_sound_effect(
                        client=el_client,
                        description=sound["description"],
                        filename=sound["suggested_filename"],
                        duration_seconds=sound.get("duration_seconds", 10.0),
                        output_dir=gen_dir,
                    )
                except Exception as e:
                    print(f"  [error] {sound.get('suggested_filename', '?')}: {e}")
        elif sounds_to_gen and not el_client:
            print(f"[warning] {len(sounds_to_gen)} sounds need generation but no ElevenLabs key set.")

        # ── Resolve paths and render ─────────────────────────────────────
        resolved_layers = resolve_layer_paths(new_plan.get("layers", []), library_dir)

        revision += 1
        timestamp = datetime.now().strftime("%H%M%S")
        out_path = str(Path(output_dir) / f"soundscape_v{revision}_{timestamp}.wav")

        print(f"\n[render] Composing revision {revision}...\n")

        try:
            render_result = compose_soundscape(
                output_path=out_path,
                duration_seconds=duration_seconds,
                layers=resolved_layers,
                master_volume_db=-1.0,
                fade_in_ms=3000,
                fade_out_ms=5000,
                sample_rate=sample_rate,
                channels=channels,
                output_format="wav",
                normalize=True,
                normalize_headroom_db=-1.0,
            )

            current_plan = new_plan
            current_path = out_path

            print(f"[done] Revision {revision}: {out_path}")
            print(f"       Layers: {render_result['layers_processed']}  "
                  f"Peak: {render_result['peak_db']} dB  "
                  f"Clipped: {render_result['clipped']}")

            # Show updated score
            score_text = plan_to_score(current_plan, duration_seconds)
            print(f"\n{score_text}\n")

            # Play it
            print(f"▶ Playing revision {revision}:\n")
            display(Audio(current_path))

        except Exception as e:
            print(f"[error] Render failed: {e}")
            import traceback
            traceback.print_exc()
            print("Keeping previous version. Try different feedback.")


# ─── Colab one-liner ────────────────────────────────────────────────────────────
# After running generate_soundscape() and getting `result`, just call:
#
#   refinement_loop(result, library_dir="/content/library", duration_seconds=180)
#
# ────────────────────────────────────────────────────────────────────────────────
