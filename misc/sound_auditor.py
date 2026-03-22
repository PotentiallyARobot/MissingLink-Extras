"""
Sound Auditor — Qwen2-Audio Quality Evaluation
================================================
Loads the Qwen2-Audio-7B-Instruct model, listens to every sound file
referenced in a .score.json, evaluates whether each sound matches its
intended description and fits the overall scene, and produces:

  1. A detailed .audit.txt report
  2. A structured .audit.json with per-file verdicts and flags

The .audit.txt is designed to be fed directly into refine_soundscape()
as additional context so the LLM knows which sounds failed and why.

Usage (Colab):
    from sound_auditor import audit_soundscape
    report = audit_soundscape(score_path="/content/output/dropship_entry.score.json")
    print(report["report_text"])

    # Then feed into refinement:
    from ai_soundscape import refine_soundscape
    result = refine_soundscape(
        instruction=open(report["report_path"]).read(),
        score_path="/content/output/dropship_entry.score.json",
    )

Requirements:
    pip install transformers librosa torch accelerate
    # Needs GPU (Colab T4/A100)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from datetime import datetime

import librosa
import numpy as np
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor


# ─── Model management ───────────────────────────────────────────────────────────

_MODEL = None
_PROCESSOR = None


def _load_model():
    """Load Qwen2-Audio model and processor. Cached after first call."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    print("[qwen-audio] Loading Qwen2-Audio-7B-Instruct...")
    t0 = time.time()

    _PROCESSOR = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    _MODEL = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"[qwen-audio] Loaded in {time.time() - t0:.1f}s")
    return _MODEL, _PROCESSOR


def _query_audio(
    audio_path: str,
    question: str,
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
    max_length: int = 512,
) -> str:
    """
    Ask Qwen2-Audio a question about a single audio file.
    Returns the model's text response.
    """
    sr = processor.feature_extractor.sampling_rate
    audio_data, _ = librosa.load(audio_path, sr=sr)

    # Truncate very long files to 30s to avoid OOM
    max_samples = sr * 30
    if len(audio_data) > max_samples:
        audio_data = audio_data[:max_samples]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "placeholder"},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    inputs = processor(
        text=text,
        audios=[audio_data],
        return_tensors="pt",
        padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generate_ids = model.generate(**inputs, max_length=max_length)

    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response.strip()


# ─── Per-file evaluation ────────────────────────────────────────────────────────

def _evaluate_sound(
    audio_path: str,
    intended_description: str,
    scene_description: str,
    label: str,
    model: Qwen2AudioForConditionalGeneration,
    processor: AutoProcessor,
) -> dict:
    """
    Evaluate a single sound file against its intended description and scene.

    Returns a dict with:
      - file, label, intended_description
      - actual_description: what Qwen hears
      - match_assessment: does it match the intention?
      - scene_fit: does it work for the scene?
      - quality_notes: any audio quality issues
      - verdict: PASS / MARGINAL / FAIL
      - flag_for_regeneration: bool
      - reason: explanation if flagged
    """
    if not Path(audio_path).exists():
        return {
            "file": audio_path,
            "label": label,
            "intended_description": intended_description,
            "actual_description": "FILE NOT FOUND",
            "match_assessment": "Cannot evaluate — file missing",
            "scene_fit": "N/A",
            "quality_notes": "File does not exist on disk",
            "verdict": "FAIL",
            "flag_for_regeneration": True,
            "reason": f"File not found: {audio_path}",
        }

    print(f"  [audit] {label}: {Path(audio_path).name}")

    # Step 1: Ask what the audio actually sounds like
    actual = _query_audio(
        audio_path,
        "Describe exactly what you hear in this audio. Be specific about "
        "the type of sound, its characteristics, texture, rhythm, and any "
        "distinct elements. Do not guess context — just describe the raw sound.",
        model, processor,
    )

    # Step 2: Ask if it matches the intended description
    match_q = (
        f'This sound was generated to be: "{intended_description}". '
        f"Based on what you actually hear, does this audio match that "
        f"description? Rate the match as STRONG MATCH, PARTIAL MATCH, "
        f"or POOR MATCH. Explain specifically what matches and what doesn't."
    )
    match_assessment = _query_audio(audio_path, match_q, model, processor)

    # Step 3: Ask about scene fit
    scene_q = (
        f'This sound is being used in a soundscape for: "{scene_description}". '
        f'It is intended to serve as: "{label}". '
        f"Does this audio work well for that purpose and scene? "
        f"Would it sound natural and convincing, or would it feel "
        f"out of place? Be specific."
    )
    scene_fit = _query_audio(audio_path, scene_q, model, processor)

    # Step 4: Audio quality check
    quality = _query_audio(
        audio_path,
        "Evaluate the audio quality of this sound. Note any issues like: "
        "distortion, clipping, unwanted artifacts, unnatural synthesis "
        "artifacts, abrupt cuts, silence, noise, or anything that would "
        "sound wrong when used in a professional sound design context. "
        "If quality is fine, say so.",
        model, processor,
    )

    # Determine verdict from the match assessment text
    match_lower = match_assessment.lower()
    quality_lower = quality.lower()
    scene_lower = scene_fit.lower()

    fail_signals = [
        "poor match", "does not match", "doesn't match", "no match",
        "completely different", "not at all", "wrong", "unrelated",
        "silence", "silent", "no sound", "empty",
    ]
    marginal_signals = [
        "partial match", "partially", "somewhat", "loosely",
        "could work", "not ideal", "with adjustments",
    ]
    quality_fail = [
        "distortion", "clipping", "artifact", "corrupt",
        "silence", "no audio", "empty", "broken",
    ]

    is_fail = any(s in match_lower for s in fail_signals)
    is_marginal = any(s in match_lower for s in marginal_signals)
    has_quality_issue = any(s in quality_lower for s in quality_fail)
    scene_poor = any(s in scene_lower for s in ["out of place", "not fit", "doesn't work", "unconvincing", "wrong"])

    if is_fail or has_quality_issue:
        verdict = "FAIL"
        flag = True
    elif is_marginal or scene_poor:
        verdict = "MARGINAL"
        flag = False  # marginal might be fixable with filters/volume
    else:
        verdict = "PASS"
        flag = False

    reason = ""
    if flag:
        reasons = []
        if is_fail:
            reasons.append("Sound does not match intended description")
        if has_quality_issue:
            reasons.append("Audio quality issues detected")
        reason = "; ".join(reasons)

    return {
        "file": audio_path,
        "label": label,
        "intended_description": intended_description,
        "actual_description": actual,
        "match_assessment": match_assessment,
        "scene_fit": scene_fit,
        "quality_notes": quality,
        "verdict": verdict,
        "flag_for_regeneration": flag,
        "reason": reason,
    }


# ─── Build file manifest from score ─────────────────────────────────────────────

def _extract_files_from_score(score: dict) -> list[dict]:
    """
    Extract unique files from the score with their descriptions and labels.
    Returns a deduplicated list.
    """
    files = {}
    gen_manifest = {
        g.get("suggested_filename", ""): g.get("description", "")
        for g in score.get("sounds_generated", [])
    }

    for layer in score.get("layers", []):
        fp = layer.get("file", "")
        if not fp or fp in files:
            continue

        # Find the intended description
        filename = Path(fp).name
        description = gen_manifest.get(filename, "")

        # If not in generated manifest, try to infer from label
        if not description:
            description = layer.get("label", filename)

        files[fp] = {
            "file": fp,
            "label": layer.get("label", ""),
            "intended_description": description,
        }

    return list(files.values())


# ─── Main audit function ────────────────────────────────────────────────────────

def audit_soundscape(
    score_path: str,
    output_dir: str | None = None,
) -> dict:
    """
    Audit all sounds in a soundscape composition using Qwen2-Audio.

    Loads the score, evaluates each unique audio file, and produces
    a report indicating which sounds match their descriptions and
    which should be regenerated.

    Parameters
    ----------
    score_path : str
        Path to the .score.json file from generate_soundscape().

    output_dir : str | None
        Where to write the report files. Defaults to same dir as the score.

    Returns
    -------
    dict with:
      - report_path: path to the .audit.txt (feed this to refine_soundscape)
      - json_path: path to the .audit.json (structured data)
      - report_text: the full report as a string
      - evaluations: list of per-file evaluation dicts
      - flagged: list of files flagged for regeneration
      - summary: quick stats string
    """
    score = json.loads(Path(score_path).read_text())
    scene = score.get("scene_description", "Unknown scene")
    duration = score.get("duration_seconds", 0)

    if output_dir is None:
        output_dir = str(Path(score_path).parent)

    # Extract files to audit
    file_manifest = _extract_files_from_score(score)
    print(f"[audit] Scene: {scene}")
    print(f"[audit] {len(file_manifest)} unique sound files to evaluate\n")

    # Load model
    model, processor = _load_model()

    # Evaluate each file
    evaluations = []
    for entry in file_manifest:
        try:
            ev = _evaluate_sound(
                audio_path=entry["file"],
                intended_description=entry["intended_description"],
                scene_description=scene,
                label=entry["label"],
                model=model,
                processor=processor,
            )
            evaluations.append(ev)
            status = ev["verdict"]
            icon = {"PASS": "OK", "MARGINAL": "~~", "FAIL": "XX"}[status]
            print(f"    [{icon}] {status}: {ev['label']}")
        except Exception as e:
            print(f"    [!!] ERROR evaluating {entry['file']}: {e}")
            evaluations.append({
                "file": entry["file"],
                "label": entry["label"],
                "intended_description": entry["intended_description"],
                "actual_description": f"ERROR: {e}",
                "match_assessment": "Could not evaluate",
                "scene_fit": "Could not evaluate",
                "quality_notes": f"Evaluation error: {e}",
                "verdict": "FAIL",
                "flag_for_regeneration": True,
                "reason": f"Evaluation error: {e}",
            })

    # Separate by verdict
    passed = [e for e in evaluations if e["verdict"] == "PASS"]
    marginal = [e for e in evaluations if e["verdict"] == "MARGINAL"]
    failed = [e for e in evaluations if e["verdict"] == "FAIL"]
    flagged = [e for e in evaluations if e["flag_for_regeneration"]]

    # ── Build the .audit.txt report ──────────────────────────────────────
    # This is the file you feed to refine_soundscape() as the instruction

    lines = []
    lines.append("SOUND QUALITY AUDIT REPORT")
    lines.append("=" * 60)
    lines.append(f"Scene: {scene}")
    lines.append(f"Duration: {duration}s")
    lines.append(f"Files audited: {len(evaluations)}")
    lines.append(f"Passed: {len(passed)}  Marginal: {len(marginal)}  Failed: {len(failed)}")
    lines.append(f"Flagged for regeneration: {len(flagged)}")
    lines.append(f"Audited: {datetime.now().isoformat()}")
    lines.append("")

    if flagged:
        lines.append("-" * 60)
        lines.append("SOUNDS THAT NEED REGENERATION:")
        lines.append("The following sounds do not match their intended description")
        lines.append("or have quality issues. Please regenerate these with new,")
        lines.append("improved ElevenLabs descriptions and update the composition.")
        lines.append("-" * 60)
        lines.append("")

        for ev in flagged:
            lines.append(f"  REGENERATE: {Path(ev['file']).name}")
            lines.append(f"    Label: {ev['label']}")
            lines.append(f"    Intended: {ev['intended_description']}")
            lines.append(f"    Actually sounds like: {ev['actual_description']}")
            lines.append(f"    Match assessment: {ev['match_assessment']}")
            lines.append(f"    Quality: {ev['quality_notes']}")
            lines.append(f"    Reason: {ev['reason']}")
            lines.append("")

    if marginal:
        lines.append("-" * 60)
        lines.append("MARGINAL SOUNDS (may benefit from adjustment):")
        lines.append("These partially match but could be improved with parameter")
        lines.append("changes (filters, volume, playback rate) or regeneration.")
        lines.append("-" * 60)
        lines.append("")

        for ev in marginal:
            lines.append(f"  REVIEW: {Path(ev['file']).name}")
            lines.append(f"    Label: {ev['label']}")
            lines.append(f"    Intended: {ev['intended_description']}")
            lines.append(f"    Actually sounds like: {ev['actual_description']}")
            lines.append(f"    Scene fit: {ev['scene_fit']}")
            lines.append("")

    if passed:
        lines.append("-" * 60)
        lines.append("PASSED SOUNDS:")
        lines.append("-" * 60)
        lines.append("")

        for ev in passed:
            lines.append(f"  OK: {Path(ev['file']).name} ({ev['label']})")

    lines.append("")
    lines.append("-" * 60)
    lines.append("END OF AUDIT REPORT")

    report_text = "\n".join(lines)

    # ── Write files ──────────────────────────────────────────────────────
    base = Path(score_path).stem.replace(".score", "")
    report_path = str(Path(output_dir) / f"{base}.audit.txt")
    json_path = str(Path(output_dir) / f"{base}.audit.json")

    Path(report_path).write_text(report_text)
    print(f"\n[audit] Report: {report_path}")

    audit_json = {
        "scene_description": scene,
        "duration_seconds": duration,
        "audited_at": datetime.now().isoformat(),
        "summary": {
            "total": len(evaluations),
            "passed": len(passed),
            "marginal": len(marginal),
            "failed": len(failed),
            "flagged_for_regeneration": len(flagged),
        },
        "evaluations": evaluations,
    }
    Path(json_path).write_text(json.dumps(audit_json, indent=2))
    print(f"[audit] JSON:   {json_path}")

    summary = (
        f"{len(passed)} passed, {len(marginal)} marginal, "
        f"{len(failed)} failed, {len(flagged)} flagged for regeneration"
    )
    print(f"[audit] {summary}")

    return {
        "report_path": report_path,
        "json_path": json_path,
        "report_text": report_text,
        "evaluations": evaluations,
        "flagged": flagged,
        "summary": summary,
    }


# ─── Convenience: audit + refine in one call ────────────────────────────────────

def audit_and_refine(
    score_path: str,
    additional_instruction: str = "",
    output_path: str | None = None,
    openai_api_key: str | None = None,
    elevenlabs_api_key: str | None = None,
) -> dict:
    """
    Audit the soundscape, then automatically feed the report into
    refine_soundscape() to fix flagged sounds.

    Parameters
    ----------
    score_path : str
        Path to the .score.json.

    additional_instruction : str
        Extra instruction to append after the audit report.
        e.g. "Also make the overall mix darker and more ominous"

    Returns the result from refine_soundscape().
    """
    from ai_soundscape import refine_soundscape

    # Run audit
    audit = audit_soundscape(score_path)

    if not audit["flagged"] and not audit.get("evaluations"):
        print("[audit+refine] Nothing flagged, skipping refinement.")
        return audit

    # Build instruction from audit report
    instruction = audit["report_text"]

    if additional_instruction:
        instruction += f"\n\nADDITIONAL INSTRUCTION:\n{additional_instruction}"

    # Refine
    print(f"\n[audit+refine] Feeding audit into refinement...\n")
    result = refine_soundscape(
        instruction=instruction,
        score_path=score_path,
        output_path=output_path,
        openai_api_key=openai_api_key,
        elevenlabs_api_key=elevenlabs_api_key,
    )

    result["audit"] = audit
    return result
