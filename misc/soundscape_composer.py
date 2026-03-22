"""
Soundscape Composer
===================
A single-function API for programmatically building rich, layered soundscapes
from a library of audio files and exporting them as a single combined file.

Dependencies:
    pip install pydub numpy
    # ffmpeg must be installed for non-wav formats

Usage:
    from soundscape_composer import compose_soundscape, TrackLayer

    result = compose_soundscape(
        output_path="my_forest_night.wav",
        duration_seconds=300,
        layers=[
            TrackLayer(
                file="library/nature/rain_heavy.wav",
                volume_db=-6,
                loop=True,
                fade_in_ms=4000,
                fade_out_ms=4000,
            ),
            TrackLayer(
                file="library/birds/owl_hoot.wav",
                volume_db=-12,
                loop=False,
                occurrences=[
                    {"start_ms": 15000},
                    {"start_ms": 60000, "volume_db": -18},
                    {"start_ms": 145000},
                ],
                pan=-0.4,
            ),
            TrackLayer(
                file="library/ambient/wind_low.wav",
                volume_db=-10,
                loop=True,
                fade_in_ms=6000,
                fade_out_ms=6000,
                start_ms=0,
                end_ms=180000,
                pan=0.3,
            ),
        ],
        master_volume_db=0,
        fade_in_ms=2000,
        fade_out_ms=3000,
        sample_rate=44100,
        channels=2,
        output_format="wav",
    )
"""

from __future__ import annotations

import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from pydub import AudioSegment


# ─── Data Models ────────────────────────────────────────────────────────────────


@dataclass
class Occurrence:
    """
    A single timed placement of a non-looping sound within the soundscape.

    Attributes:
        start_ms:       When this occurrence begins (ms from soundscape start).
        volume_db:      Per-occurrence volume override (relative to track volume).
                        None means use the track's volume_db.
        pan:            Per-occurrence pan override. None means use track pan.
        fade_in_ms:     Per-occurrence fade-in override.
        fade_out_ms:    Per-occurrence fade-out override.
        playback_rate:  Speed/pitch multiplier (1.0 = normal, 0.5 = half speed,
                        2.0 = double speed). Useful for natural variation.
        trim_start_ms:  Trim the source audio from this point before placing.
        trim_end_ms:    Trim the source audio up to this point before placing.
                        None means use the full remaining audio.
        reverse:        If True, reverse this occurrence's audio.
    """
    start_ms: int = 0
    volume_db: float | None = None
    pan: float | None = None
    fade_in_ms: int | None = None
    fade_out_ms: int | None = None
    playback_rate: float = 1.0
    trim_start_ms: int = 0
    trim_end_ms: int | None = None
    reverse: bool = False


@dataclass
class AutomationPoint:
    """
    A single keyframe in a volume or pan automation envelope.

    Attributes:
        time_ms:  Position in the soundscape timeline (ms).
        value:    The target value at this point.
                  For volume automation: value in dB (e.g., -6.0).
                  For pan automation: value from -1.0 (left) to 1.0 (right).
    """
    time_ms: int = 0
    value: float = 0.0


@dataclass
class TrackLayer:
    """
    One audio layer in the soundscape.

    Core:
        file:               Path to the source audio file.
        label:              Optional human-readable label for this track.

    Timing:
        start_ms:           When this layer begins in the output (ms). Default 0.
        end_ms:             When this layer ends (ms). None = end of soundscape.
                            Audio is truncated/looped to fit this window.

    Volume & Panning:
        volume_db:          Gain adjustment in dB (0 = original, -6 = half, +6 = double).
        pan:                Stereo position from -1.0 (hard left) to 1.0 (hard right).
                            0.0 = center.

    Looping:
        loop:               If True, the audio loops continuously within its window.
        loop_crossfade_ms:  Crossfade duration at loop boundaries to eliminate clicks.
                            Only used when loop=True.

    Fading:
        fade_in_ms:         Fade-in duration at the start of the layer.
        fade_out_ms:        Fade-out duration at the end of the layer.
        fade_in_curve:       Shape of fade-in: "linear", "logarithmic", "exponential",
                            or "scurve".
        fade_out_curve:     Shape of fade-out (same options).

    Trimming & Manipulation:
        trim_start_ms:      Trim source audio from this point.
        trim_end_ms:        Trim source audio up to this point.
        playback_rate:      Speed/pitch multiplier.
        reverse:            Reverse the source audio before processing.

    Scheduling (for one-shot / sporadic sounds):
        occurrences:        List of Occurrence dicts/objects for precise placement.
                            If provided, the sound is placed at each specified time
                            instead of as a continuous/looping layer.
        random_occurrences: Auto-generate random placements.
                            Dict with keys:
                                count:          Number of placements.
                                min_gap_ms:     Minimum gap between placements.
                                volume_var_db:  Random volume variation +/- dB.
                                pan_var:        Random pan variation +/-.
                                rate_var:       Random playback rate variation +/-.

    Automation:
        volume_automation:  List of AutomationPoint for volume changes over time.
                            Interpolated linearly between points. Overrides static
                            volume_db where defined.
        pan_automation:     List of AutomationPoint for panning changes over time.

    Effects:
        low_pass_hz:        Apply a simple low-pass filter at this frequency.
                            None = no filter. Useful for "distant" sounds.
        high_pass_hz:       Apply a simple high-pass filter at this frequency.
                            None = no filter. Useful for removing rumble.
    """

    # Core
    file: str = ""
    label: str = ""

    # Timing
    start_ms: int = 0
    end_ms: int | None = None

    # Volume & pan
    volume_db: float = 0.0
    pan: float = 0.0

    # Looping
    loop: bool = False
    loop_crossfade_ms: int = 100

    # Fading
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    fade_in_curve: Literal["linear", "logarithmic", "exponential", "scurve"] = "linear"
    fade_out_curve: Literal["linear", "logarithmic", "exponential", "scurve"] = "linear"

    # Trimming & manipulation
    trim_start_ms: int = 0
    trim_end_ms: int | None = None
    playback_rate: float = 1.0
    reverse: bool = False

    # Scheduling
    occurrences: list[dict | Occurrence] | None = None
    random_occurrences: dict | None = None

    # Automation
    volume_automation: list[dict | AutomationPoint] | None = None
    pan_automation: list[dict | AutomationPoint] | None = None

    # Effects
    low_pass_hz: int | None = None
    high_pass_hz: int | None = None


# ─── Internal Helpers ───────────────────────────────────────────────────────────


def _load_audio(file_path: str, sample_rate: int, channels: int) -> AudioSegment:
    """Load an audio file and normalize to target sample rate and channels."""
    seg = AudioSegment.from_file(file_path)
    seg = seg.set_frame_rate(sample_rate)
    seg = seg.set_channels(channels)
    return seg


def _trim(seg: AudioSegment, start_ms: int, end_ms: int | None) -> AudioSegment:
    """Trim an audio segment."""
    if end_ms is not None:
        return seg[start_ms:end_ms]
    return seg[start_ms:]


def _change_speed(seg: AudioSegment, rate: float) -> AudioSegment:
    """Change playback speed by resampling (also changes pitch)."""
    if abs(rate - 1.0) < 0.01:
        return seg
    new_rate = int(seg.frame_rate * rate)
    modified = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
    return modified.set_frame_rate(seg.frame_rate)


def _apply_fade_curve(
    seg: AudioSegment,
    duration_ms: int,
    curve: str,
    is_fade_in: bool,
) -> AudioSegment:
    """Apply a fade with a specific curve shape."""
    if duration_ms <= 0:
        return seg

    if curve == "linear":
        if is_fade_in:
            return seg.fade_in(duration_ms)
        return seg.fade_out(duration_ms)

    # For non-linear curves, we process sample-by-sample via numpy
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n_channels = seg.channels
    n_fade_samples = int(duration_ms * seg.frame_rate / 1000) * n_channels

    if n_fade_samples > len(samples):
        n_fade_samples = len(samples)

    t = np.linspace(0, 1, n_fade_samples)

    if curve == "logarithmic":
        envelope = np.log1p(t * (math.e - 1)) # 0→1 log curve
    elif curve == "exponential":
        envelope = (np.exp(t) - 1) / (math.e - 1)
    elif curve == "scurve":
        envelope = 0.5 * (1 - np.cos(np.pi * t))
    else:
        envelope = t  # fallback linear

    if not is_fade_in:
        envelope = envelope[::-1]

    all_samples = samples.copy()
    if is_fade_in:
        all_samples[:n_fade_samples] *= envelope
    else:
        all_samples[-n_fade_samples:] *= envelope

    all_samples = np.clip(all_samples, -32768, 32767).astype(np.int16)
    return seg._spawn(all_samples.tobytes())


def _loop_to_length(seg: AudioSegment, target_ms: int, crossfade_ms: int) -> AudioSegment:
    """Loop audio to fill the target duration with optional crossfading at seams."""
    if len(seg) == 0:
        return AudioSegment.silent(duration=target_ms, frame_rate=seg.frame_rate)

    if len(seg) >= target_ms:
        return seg[:target_ms]

    crossfade = min(crossfade_ms, len(seg) // 2, 50)
    crossfade = max(crossfade, 0)

    result = seg
    while len(result) < target_ms:
        remaining = target_ms - len(result)
        chunk = seg[:remaining + crossfade] if remaining + crossfade < len(seg) else seg
        if crossfade > 0 and len(result) > crossfade and len(chunk) > crossfade:
            result = result.append(chunk, crossfade=crossfade)
        else:
            result = result + chunk

    return result[:target_ms]


def _apply_pan(seg: AudioSegment, pan: float) -> AudioSegment:
    """Apply stereo panning. -1.0 = left, 0.0 = center, 1.0 = right."""
    pan = max(-1.0, min(1.0, pan))
    if abs(pan) < 0.01:
        return seg
    return seg.pan(pan)


def _apply_low_pass(seg: AudioSegment, cutoff_hz: int) -> AudioSegment:
    """Simple single-pole low-pass filter via numpy."""
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n_ch = seg.channels
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / seg.frame_rate
    alpha = dt / (rc + dt)

    # Process each channel independently
    reshaped = samples.reshape(-1, n_ch)
    for ch in range(n_ch):
        channel = reshaped[:, ch]
        for i in range(1, len(channel)):
            channel[i] = channel[i - 1] + alpha * (channel[i] - channel[i - 1])
        reshaped[:, ch] = channel

    out = np.clip(reshaped.flatten(), -32768, 32767).astype(np.int16)
    return seg._spawn(out.tobytes())


def _apply_high_pass(seg: AudioSegment, cutoff_hz: int) -> AudioSegment:
    """Simple single-pole high-pass filter via numpy."""
    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n_ch = seg.channels
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / seg.frame_rate
    alpha = rc / (rc + dt)

    reshaped = samples.reshape(-1, n_ch)
    for ch in range(n_ch):
        channel = reshaped[:, ch].copy()
        prev_raw = channel[0]
        for i in range(1, len(channel)):
            raw = channel[i]
            channel[i] = alpha * (channel[i - 1] + raw - prev_raw)
            prev_raw = raw
        reshaped[:, ch] = channel

    out = np.clip(reshaped.flatten(), -32768, 32767).astype(np.int16)
    return seg._spawn(out.tobytes())


def _apply_volume_automation(
    seg: AudioSegment,
    points: list[AutomationPoint],
    layer_start_ms: int,
) -> AudioSegment:
    """Apply volume automation envelope to a segment."""
    if not points or len(seg) == 0:
        return seg

    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n_ch = seg.channels
    n_frames = len(samples) // n_ch
    frame_rate = seg.frame_rate

    # Build an envelope array (one value per frame)
    envelope_db = np.zeros(n_frames)
    sorted_pts = sorted(points, key=lambda p: p.time_ms)

    # Convert automation times to frame indices relative to this segment
    keyframes = []
    for p in sorted_pts:
        frame_idx = int((p.time_ms - layer_start_ms) * frame_rate / 1000)
        frame_idx = max(0, min(n_frames - 1, frame_idx))
        keyframes.append((frame_idx, p.value))

    # Fill envelope by linear interpolation
    if len(keyframes) == 1:
        envelope_db[:] = keyframes[0][1]
    else:
        # Before first keyframe
        envelope_db[: keyframes[0][0]] = keyframes[0][1]
        # Between keyframes
        for i in range(len(keyframes) - 1):
            f0, v0 = keyframes[i]
            f1, v1 = keyframes[i + 1]
            if f1 > f0:
                envelope_db[f0:f1] = np.linspace(v0, v1, f1 - f0)
            else:
                envelope_db[f0] = v0
        # After last keyframe
        envelope_db[keyframes[-1][0] :] = keyframes[-1][1]

    # Convert dB envelope to linear gain and apply
    gain = np.power(10, envelope_db / 20.0)
    gain_per_sample = np.repeat(gain, n_ch)
    samples *= gain_per_sample
    out = np.clip(samples, -32768, 32767).astype(np.int16)
    return seg._spawn(out.tobytes())


def _apply_pan_automation(
    seg: AudioSegment,
    points: list[AutomationPoint],
    layer_start_ms: int,
) -> AudioSegment:
    """Apply pan automation to a stereo segment."""
    if not points or len(seg) == 0 or seg.channels != 2:
        return seg

    samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
    n_frames = len(samples) // 2
    frame_rate = seg.frame_rate

    sorted_pts = sorted(points, key=lambda p: p.time_ms)
    keyframes = []
    for p in sorted_pts:
        frame_idx = int((p.time_ms - layer_start_ms) * frame_rate / 1000)
        frame_idx = max(0, min(n_frames - 1, frame_idx))
        keyframes.append((frame_idx, max(-1.0, min(1.0, p.value))))

    pan_env = np.zeros(n_frames)
    if len(keyframes) == 1:
        pan_env[:] = keyframes[0][1]
    else:
        pan_env[: keyframes[0][0]] = keyframes[0][1]
        for i in range(len(keyframes) - 1):
            f0, v0 = keyframes[i]
            f1, v1 = keyframes[i + 1]
            if f1 > f0:
                pan_env[f0:f1] = np.linspace(v0, v1, f1 - f0)
        pan_env[keyframes[-1][0] :] = keyframes[-1][1]

    # Equal-power pan law
    reshaped = samples.reshape(-1, 2)
    left_gain = np.cos((pan_env + 1) * math.pi / 4)
    right_gain = np.sin((pan_env + 1) * math.pi / 4)
    # Apply relative to center (which gives ~0.707 for each channel)
    center = math.cos(math.pi / 4)
    reshaped[:, 0] *= left_gain / center
    reshaped[:, 1] *= right_gain / center

    out = np.clip(reshaped.flatten(), -32768, 32767).astype(np.int16)
    return seg._spawn(out.tobytes())


def _coerce_occurrence(o: dict | Occurrence) -> Occurrence:
    if isinstance(o, Occurrence):
        return o
    import dataclasses as _dc
    valid_keys = {f.name for f in _dc.fields(Occurrence)}
    cleaned = {k: v for k, v in o.items() if k in valid_keys}
    return Occurrence(**cleaned)


def _coerce_automation_point(p: dict | AutomationPoint) -> AutomationPoint:
    if isinstance(p, AutomationPoint):
        return p
    d = dict(p)
    # GPT sometimes outputs "value_db" or "value_pan" instead of "value"
    if "value" not in d:
        for alias in ("value_db", "value_pan", "db", "pan_value"):
            if alias in d:
                d["value"] = d.pop(alias)
                break
    # Strip any keys that AutomationPoint doesn't accept
    valid_keys = {"time_ms", "value"}
    d = {k: v for k, v in d.items() if k in valid_keys}
    return AutomationPoint(**d)


# ─── Main Compose Function ─────────────────────────────────────────────────────


def compose_soundscape(
    output_path: str,
    duration_seconds: float,
    layers: list[TrackLayer | dict],
    master_volume_db: float = 0.0,
    fade_in_ms: int = 0,
    fade_out_ms: int = 0,
    sample_rate: int = 44100,
    channels: int = 2,
    output_format: str = "wav",
    bit_depth: int = 16,
    normalize: bool = False,
    normalize_headroom_db: float = -1.0,
) -> dict:
    """
    Compose a multi-layered soundscape and export to a single audio file.

    Parameters
    ----------
    output_path : str
        Output file path (e.g., "output/forest_night.wav").

    duration_seconds : float
        Total duration of the output soundscape in seconds.

    layers : list[TrackLayer | dict]
        List of audio layers to mix. Each can be a TrackLayer instance or
        a dict with the same keys. See TrackLayer docstring for all options.

    master_volume_db : float
        Master gain applied to the final mix (default 0.0 = unity).

    fade_in_ms : int
        Master fade-in applied to the final mix.

    fade_out_ms : int
        Master fade-out applied to the final mix.

    sample_rate : int
        Output sample rate in Hz (default 44100).

    channels : int
        Output channel count (1 = mono, 2 = stereo; default 2).

    output_format : str
        Export format: "wav", "mp3", "ogg", "flac" (default "wav").

    bit_depth : int
        Bit depth for wav/flac output: 16 or 24 (default 16).

    normalize : bool
        If True, normalize the final mix to prevent clipping.

    normalize_headroom_db : float
        Target peak level when normalizing (default -1.0 dB).

    Returns
    -------
    dict with keys:
        - path: str — the output file path
        - duration_ms: int — actual output duration
        - layers_processed: int — how many layers were rendered
        - peak_db: float — peak level of the final mix
        - clipped: bool — whether the mix clipped before export
    """

    total_ms = int(duration_seconds * 1000)

    # Create silent canvas
    canvas = AudioSegment.silent(duration=total_ms, frame_rate=sample_rate)
    canvas = canvas.set_channels(channels)

    layers_processed = 0

    for layer_input in layers:
        # Accept dicts or TrackLayer instances
        if isinstance(layer_input, dict):
            import dataclasses as _dc
            valid_keys = {f.name for f in _dc.fields(TrackLayer)}
            cleaned_input = {k: v for k, v in layer_input.items() if k in valid_keys}
            layer = TrackLayer(**cleaned_input)
        else:
            layer = layer_input

        if not layer.file:
            warnings.warn(f"Skipping layer with no file specified (label='{layer.label}')")
            continue

        if not Path(layer.file).exists():
            warnings.warn(f"File not found, skipping: {layer.file}")
            continue

        # Load source audio
        source = _load_audio(layer.file, sample_rate, channels)

        # Trim source
        source = _trim(source, layer.trim_start_ms, layer.trim_end_ms)

        # Reverse
        if layer.reverse:
            source = source.reverse()

        # Playback rate
        source = _change_speed(source, layer.playback_rate)

        # Determine the layer's time window
        layer_start = max(0, layer.start_ms)
        layer_end = min(total_ms, layer.end_ms if layer.end_ms is not None else total_ms)
        window_ms = layer_end - layer_start

        if window_ms <= 0:
            continue

        # ── Occurrence-based placement (one-shots / sporadic sounds) ─────
        if layer.occurrences is not None or layer.random_occurrences is not None:
            all_occurrences: list[Occurrence] = []

            # Explicit occurrences
            if layer.occurrences:
                all_occurrences.extend(_coerce_occurrence(o) for o in layer.occurrences)

            # Random occurrences
            if layer.random_occurrences:
                rc = layer.random_occurrences
                count = rc.get("count", 5)
                min_gap = rc.get("min_gap_ms", 2000)
                vol_var = rc.get("volume_var_db", 3.0)
                pan_var = rc.get("pan_var", 0.2)
                rate_var = rc.get("rate_var", 0.0)

                rng = np.random.default_rng()
                times = []
                for _ in range(count * 10):  # attempts
                    t = int(rng.uniform(layer_start, max(layer_start + 1, layer_end - len(source))))
                    if all(abs(t - existing) >= min_gap for existing in times):
                        times.append(t)
                    if len(times) >= count:
                        break

                for t in times:
                    all_occurrences.append(Occurrence(
                        start_ms=t,
                        volume_db=rng.uniform(-vol_var, vol_var) if vol_var else None,
                        pan=layer.pan + rng.uniform(-pan_var, pan_var) if pan_var else None,
                        playback_rate=layer.playback_rate + rng.uniform(-rate_var, rate_var) if rate_var else 1.0,
                    ))

            # Place each occurrence
            for occ in all_occurrences:
                seg = source

                # Per-occurrence trim
                if occ.trim_start_ms or occ.trim_end_ms:
                    seg = _trim(seg, occ.trim_start_ms, occ.trim_end_ms)

                # Per-occurrence speed
                if abs(occ.playback_rate - 1.0) > 0.01:
                    seg = _change_speed(seg, occ.playback_rate)

                # Per-occurrence reverse
                if occ.reverse:
                    seg = seg.reverse()

                # Volume
                vol = occ.volume_db if occ.volume_db is not None else layer.volume_db
                seg = seg + vol  # pydub: segment + dB

                # Fades
                fi = occ.fade_in_ms if occ.fade_in_ms is not None else layer.fade_in_ms
                fo = occ.fade_out_ms if occ.fade_out_ms is not None else layer.fade_out_ms
                seg = _apply_fade_curve(seg, fi, layer.fade_in_curve, is_fade_in=True)
                seg = _apply_fade_curve(seg, fo, layer.fade_out_curve, is_fade_in=False)

                # Pan
                p = occ.pan if occ.pan is not None else layer.pan
                seg = _apply_pan(seg, p)

                # Filters
                if layer.low_pass_hz:
                    seg = _apply_low_pass(seg, layer.low_pass_hz)
                if layer.high_pass_hz:
                    seg = _apply_high_pass(seg, layer.high_pass_hz)

                # Overlay onto canvas
                pos = max(0, occ.start_ms)
                if pos < total_ms:
                    canvas = canvas.overlay(seg, position=pos)

        # ── Continuous / looping layer ───────────────────────────────────
        else:
            # Build the segment to fill the window
            if layer.loop:
                seg = _loop_to_length(source, window_ms, layer.loop_crossfade_ms)
            else:
                seg = source[:window_ms]

            # Pad if shorter than window and not looping
            if len(seg) < window_ms and not layer.loop:
                seg = seg + AudioSegment.silent(
                    duration=window_ms - len(seg),
                    frame_rate=sample_rate,
                )

            # Volume
            seg = seg + layer.volume_db

            # Fades
            seg = _apply_fade_curve(seg, layer.fade_in_ms, layer.fade_in_curve, is_fade_in=True)
            seg = _apply_fade_curve(seg, layer.fade_out_ms, layer.fade_out_curve, is_fade_in=False)

            # Filters
            if layer.low_pass_hz:
                seg = _apply_low_pass(seg, layer.low_pass_hz)
            if layer.high_pass_hz:
                seg = _apply_high_pass(seg, layer.high_pass_hz)

            # Volume automation
            if layer.volume_automation:
                pts = [_coerce_automation_point(p) for p in layer.volume_automation]
                seg = _apply_volume_automation(seg, pts, layer_start)

            # Pan (static or automated)
            if layer.pan_automation:
                pts = [_coerce_automation_point(p) for p in layer.pan_automation]
                seg = _apply_pan_automation(seg, pts, layer_start)
            else:
                seg = _apply_pan(seg, layer.pan)

            # Overlay onto canvas
            canvas = canvas.overlay(seg, position=layer_start)

        layers_processed += 1

    # ── Master processing ───────────────────────────────────────────────────────

    # Master volume
    if master_volume_db != 0.0:
        canvas = canvas + master_volume_db

    # Detect clipping
    peak = canvas.max
    peak_db = 20 * math.log10(peak / 32768) if peak > 0 else -math.inf
    clipped = peak >= 32767

    # Normalize
    if normalize and peak > 0:
        target_peak = 32768 * (10 ** (normalize_headroom_db / 20))
        gain_needed = 20 * math.log10(target_peak / peak)
        canvas = canvas + gain_needed
        peak_db = normalize_headroom_db
        clipped = False

    # Master fades
    if fade_in_ms > 0:
        canvas = canvas.fade_in(min(fade_in_ms, total_ms))
    if fade_out_ms > 0:
        canvas = canvas.fade_out(min(fade_out_ms, total_ms))

    # ── Export ──────────────────────────────────────────────────────────────────

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    export_params = {}
    if output_format in ("wav", "flac") and bit_depth == 24:
        export_params["sample_width"] = 3

    canvas.export(output_path, format=output_format, **export_params)

    return {
        "path": str(output_path),
        "duration_ms": len(canvas),
        "layers_processed": layers_processed,
        "peak_db": round(peak_db, 2),
        "clipped": clipped,
    }


# ─── Convenience: List available library files ──────────────────────────────────


def list_library(library_dir: str = "library") -> list[dict]:
    """
    Scan a library directory and return metadata for all audio files found.
    Useful for an AI to know what sounds are available.

    Returns a list of dicts with keys: path, name, folder, ext, duration_ms, channels.
    """
    AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".webm"}
    root = Path(library_dir)
    results = []
    if not root.exists():
        return results
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            try:
                seg = AudioSegment.from_file(str(f))
                results.append({
                    "path": str(f),
                    "name": f.stem,
                    "folder": str(f.parent.relative_to(root)),
                    "ext": f.suffix.lower(),
                    "duration_ms": len(seg),
                    "channels": seg.channels,
                    "sample_rate": seg.frame_rate,
                })
            except Exception:
                results.append({
                    "path": str(f),
                    "name": f.stem,
                    "folder": str(f.parent.relative_to(root)),
                    "ext": f.suffix.lower(),
                    "duration_ms": None,
                    "channels": None,
                    "sample_rate": None,
                })
    return results


# ─── Example usage ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # This example shows the full API surface.
    # Replace file paths with your actual library files.

    result = compose_soundscape(
        output_path="output/demo_soundscape.wav",
        duration_seconds=120,
        layers=[
            # Continuous rain loop as the base
            TrackLayer(
                file="library/weather/rain_medium.wav",
                label="Rain base",
                volume_db=-4,
                loop=True,
                loop_crossfade_ms=200,
                fade_in_ms=5000,
                fade_out_ms=5000,
                fade_in_curve="scurve",
            ),
            # Wind that pans slowly left to right
            TrackLayer(
                file="library/weather/wind_gentle.wav",
                label="Wind sweep",
                volume_db=-8,
                loop=True,
                pan_automation=[
                    {"time_ms": 0,      "value": -0.6},
                    {"time_ms": 30000,  "value": 0.6},
                    {"time_ms": 60000,  "value": -0.6},
                    {"time_ms": 90000,  "value": 0.6},
                    {"time_ms": 120000, "value": -0.6},
                ],
            ),
            # Distant thunder — randomly placed
            TrackLayer(
                file="library/weather/thunder_distant.wav",
                label="Distant thunder",
                volume_db=-14,
                low_pass_hz=800,
                random_occurrences={
                    "count": 4,
                    "min_gap_ms": 15000,
                    "volume_var_db": 4,
                    "pan_var": 0.5,
                },
            ),
            # Owl hoots at specific moments
            TrackLayer(
                file="library/animals/owl_hoot.wav",
                label="Owl",
                volume_db=-10,
                occurrences=[
                    {"start_ms": 20000, "pan": -0.5},
                    {"start_ms": 55000, "pan": 0.3, "volume_db": -14},
                    {"start_ms": 95000, "pan": -0.7},
                ],
            ),
            # Campfire crackle that fades in late, panned slightly right
            TrackLayer(
                file="library/ambience/campfire.wav",
                label="Campfire",
                volume_db=-6,
                loop=True,
                start_ms=30000,
                fade_in_ms=8000,
                fade_in_curve="logarithmic",
                fade_out_ms=10000,
                pan=0.25,
            ),
        ],
        master_volume_db=-1,
        fade_in_ms=3000,
        fade_out_ms=5000,
        sample_rate=44100,
        channels=2,
        output_format="wav",
        normalize=True,
        normalize_headroom_db=-1.0,
    )

    print(f"Exported: {result['path']}")
    print(f"Duration: {result['duration_ms'] / 1000:.1f}s")
    print(f"Layers:   {result['layers_processed']}")
    print(f"Peak:     {result['peak_db']} dB")
    print(f"Clipped:  {result['clipped']}")