import json
import time

import gradio as gr
import requests

from deploy_twilio_from_colab import DEPLOY_INFO_PATH, DeployError, deploy


def _load_deploy_info():
    if DEPLOY_INFO_PATH.exists():
        try:
            return json.loads(DEPLOY_INFO_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "base_url": "",
        "admin_key": "",
        "sync_service_sid": "",
        "environment": "dev",
    }


def _ensure_backend():
    info = _load_deploy_info()
    base_url = (info.get("base_url") or "").strip().rstrip("/")
    admin_key = (info.get("admin_key") or "").strip()

    if base_url and admin_key:
        return info, "Backend ready."

    info = deploy()
    return info, "Backend deployed automatically."


def _safe_json(resp):
    try:
        return resp.json()
    except Exception:
        text = getattr(resp, "text", "")
        return {"ok": False, "error": f"Non-JSON response: {text[:1000]}"}


def _turn_text(turn):
    return (
        turn.get("text")
        or turn.get("content")
        or turn.get("message")
        or turn.get("utterance")
        or ""
    ).strip()


def _turn_role(turn):
    return (
        turn.get("role")
        or turn.get("speaker")
        or turn.get("from")
        or "unknown"
    ).strip().upper()


def _turn_time(turn):
    return (
        turn.get("at")
        or turn.get("timestamp")
        or turn.get("time")
        or ""
    ).strip()


def _format_transcript(turns):
    if not turns:
        return "Waiting for transcript..."
    parts = []
    for turn in turns:
        ts = _turn_time(turn)
        role = _turn_role(turn)
        text = _turn_text(turn)
        prefix = f"[{ts}] " if ts else ""
        parts.append(f"{prefix}{role}: {text}")
    return "\n\n".join(parts)


def _format_status(data):
    status = data.get("status", "unknown")
    turn_count = data.get("turn_count", len(data.get("turns", []) or []))
    max_turns = data.get("max_turns", "")
    call_sid = data.get("call_sid", "")
    confidence = data.get("confidence", "")
    error = data.get("error", "")

    lines = [f"Status: {status}"]

    if max_turns != "":
        lines.append(f"Turns: {turn_count}/{max_turns}")
    else:
        lines.append(f"Turns: {turn_count}")

    if confidence:
        lines.append(f"Confidence: {confidence}")
    if call_sid:
        lines.append(f"Call SID: {call_sid}")
    if error:
        lines.append(f"Error: {error}")

    return "\n".join(lines)


def _extract_final_answer(data):
    for key in ["result", "final_answer", "answer", "summary", "final_result"]:
        value = (data.get(key) or "").strip() if isinstance(data.get(key), str) else data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def start_agent_call(to_number, objective, contact_name, max_turns):
    to_number = (to_number or "").strip()
    objective = (objective or "").strip()
    contact_name = (contact_name or "").strip()
    max_turns = int(max_turns or 6)

    if not to_number:
        yield "Enter a phone number.", "", "", ""
        return

    if not objective:
        yield "Enter instructions for the agent.", "", "", ""
        return

    yield "Preparing backend...", "", "", ""

    try:
        info, prep_msg = _ensure_backend()
    except DeployError as e:
        yield f"Backend deployment failed:\n\n{e}", "", "", ""
        return
    except Exception as e:
        yield f"Backend deployment failed unexpectedly:\n\n{e}", "", "", ""
        return

    base_url = (info.get("base_url") or "").strip().rstrip("/")
    admin_key = (info.get("admin_key") or "").strip()

    if not base_url or not admin_key:
        yield "Backend setup did not produce a usable base URL/admin key.", "", "", ""
        return

    yield prep_msg + "\n\nStarting call...", "", "", ""

    try:
        resp = requests.post(
            f"{base_url}/start",
            data={
                "admin_key": admin_key,
                "to": to_number,
                "objective": objective,
                "contact_name": contact_name,
                "max_turns": str(max_turns),
            },
            timeout=60,
        )
        data = _safe_json(resp)
    except Exception as e:
        yield f"Call start failed:\n\n{e}", "", "", ""
        return

    if not resp.ok or not data.get("ok"):
        yield "Call start failed:\n\n" + json.dumps(data, indent=2), "", "", ""
        return

    job_id = data["job_id"]
    admin_token = data["admin_token"]

    yield f"Call started.\nJob ID: {job_id}\nConnecting...", "", "", job_id

    last_transcript = ""
    last_result = ""

    for _ in range(240):  # about 8 minutes at 2s polling
        try:
            resp = requests.get(
                f"{base_url}/status",
                params={"job_id": job_id, "admin_token": admin_token},
                timeout=30,
            )
            data = _safe_json(resp)
        except Exception as e:
            yield f"Refresh failed: {e}", last_transcript, last_result, job_id
            time.sleep(2)
            continue

        if not resp.ok or not data.get("ok"):
            yield "Status error:\n\n" + json.dumps(data, indent=2), last_transcript, last_result, job_id
            time.sleep(2)
            continue

        transcript = _format_transcript(data.get("turns", []) or [])
        result = _extract_final_answer(data)
        status_text = _format_status(data)

        last_transcript = transcript
        last_result = result

        yield status_text, transcript, result, job_id

        if str(data.get("status", "")).lower() in {"done", "completed", "complete", "failed", "error"}:
            return

        time.sleep(2)

    yield "Timed out waiting for completion.", last_transcript, last_result, job_id


def build_ui():
    with gr.Blocks(title="Phone Agent") as demo:
        gr.Markdown("# Phone Agent")
        gr.Markdown(
            "Enter a phone number and the agent's goal. "
            "The backend will be deployed automatically from your Colab secrets if needed."
        )

        with gr.Row():
            to_number = gr.Textbox(
                label="Phone number",
                placeholder="+12025550123",
                scale=1,
            )
            contact_name = gr.Textbox(
                label="Contact name (optional)",
                placeholder="Tyler",
                scale=1,
            )

        objective = gr.Textbox(
            label="Instructions for the agent",
            lines=6,
            placeholder=(
                "Call Tyler and find out where he put his coat. "
                "When you have the information, say thank you, goodbye, and hang up."
            ),
        )

        with gr.Accordion("Advanced", open=False):
            max_turns = gr.Slider(
                label="Max turns",
                minimum=2,
                maximum=12,
                step=1,
                value=6,
            )

        start_btn = gr.Button("Start agent call", variant="primary")

        status_box = gr.Textbox(label="Status", lines=6)
        transcript_box = gr.Textbox(label="Live transcript", lines=18)
        final_answer_box = gr.Textbox(label="Final answer", lines=4)
        job_id_box = gr.Textbox(label="Job ID", visible=False)

        start_btn.click(
            start_agent_call,
            inputs=[to_number, objective, contact_name, max_turns],
            outputs=[status_box, transcript_box, final_answer_box, job_id_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        share=False,
        inline=True,
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
    )