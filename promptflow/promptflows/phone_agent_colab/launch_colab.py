import json
import os
from pathlib import Path

import gradio as gr
import requests

from deploy_twilio_from_colab import DEPLOY_INFO_PATH, DeployError, deploy

ROOT = Path(__file__).resolve().parent


def _load_deploy_info():
    if DEPLOY_INFO_PATH.exists():
        return json.loads(DEPLOY_INFO_PATH.read_text(encoding="utf-8"))
    return {
        "base_url": "",
        "admin_key": "",
        "sync_service_sid": "",
        "environment": "dev",
    }


def deploy_backend():
    try:
        info = deploy()
        msg = (
            f"Backend deployed.\n\n"
            f"Base URL: {info['base_url']}\n"
            f"Admin key saved locally in deployment.json\n"
            f"Sync service: {info['sync_service_sid']}"
        )
        return info["base_url"], info["admin_key"], msg
    except DeployError as e:
        return gr.update(), gr.update(), f"Deploy failed:\n\n{e}"
    except Exception as e:
        return gr.update(), gr.update(), f"Deploy failed with an unexpected error:\n\n{e}"


def start_call(base_url, admin_key, to_number, objective, contact_name, max_turns):
    base_url = (base_url or "").strip().rstrip("/")
    admin_key = (admin_key or "").strip()
    to_number = (to_number or "").strip()
    objective = (objective or "").strip()
    contact_name = (contact_name or "").strip()

    if not base_url or not admin_key:
        return "", "", "Please deploy the backend first or paste the base URL and admin key."
    if not to_number or not objective:
        return "", "", "Please enter both a destination number and an objective."

    try:
        resp = requests.post(
            f"{base_url}/start",
            data={
                "admin_key": admin_key,
                "to": to_number,
                "objective": objective,
                "contact_name": contact_name,
                "max_turns": str(int(max_turns or 6)),
            },
            timeout=45,
        )
        data = resp.json()
    except Exception as e:
        return "", "", f"Call start failed: {e}"

    if not resp.ok or not data.get("ok"):
        return "", "", json.dumps(data, indent=2)

    job_id = data["job_id"]
    token = data["admin_token"]
    pretty = json.dumps(data, indent=2)
    return job_id, token, pretty


def refresh_job(base_url, job_id, token):
    base_url = (base_url or "").strip().rstrip("/")
    job_id = (job_id or "").strip()
    token = (token or "").strip()
    if not base_url or not job_id or not token:
        return "Paste or create a job first."

    try:
        resp = requests.get(
            f"{base_url}/status",
            params={"job_id": job_id, "admin_token": token},
            timeout=30,
        )
        data = resp.json()
    except Exception as e:
        return f"Refresh failed: {e}"

    return json.dumps(data, indent=2)


def build_ui():
    info = _load_deploy_info()

    with gr.Blocks(title="Colab Phone Agent") as demo:
        gr.Markdown("# Colab Phone Agent")
        gr.Markdown(
            "Deploy the Twilio Functions backend once, then enter a phone number and a goal. "
            "The agent will place the call, pursue the objective, and store the final extracted answer."
        )

        with gr.Tab("1. Deploy backend"):
            deploy_btn = gr.Button("Deploy / refresh backend", variant="primary")
            base_url = gr.Textbox(label="Twilio Functions base URL", value=info.get("base_url", ""))
            admin_key = gr.Textbox(label="Admin key", value=info.get("admin_key", ""), type="password")
            deploy_log = gr.Textbox(label="Deploy log", lines=10)
            deploy_btn.click(deploy_backend, outputs=[base_url, admin_key, deploy_log])

        with gr.Tab("2. Start call"):
            to_number = gr.Textbox(label="Destination number (E.164)", placeholder="+12025550123")
            contact_name = gr.Textbox(label="Optional contact name", placeholder="Tyler")
            objective = gr.Textbox(
                label="Objective",
                lines=5,
                placeholder=(
                    "Call Tyler and find out where he put his coat. "
                    "When you have the information, say thank you, goodbye, and hang up."
                ),
            )
            max_turns = gr.Slider(label="Max turns", minimum=2, maximum=12, step=1, value=6)
            start_btn = gr.Button("Start call", variant="primary")
            job_id = gr.Textbox(label="Job ID")
            job_token = gr.Textbox(label="Job token", type="password")
            start_result = gr.Textbox(label="Start result", lines=12)
            start_btn.click(
                start_call,
                inputs=[base_url, admin_key, to_number, objective, contact_name, max_turns],
                outputs=[job_id, job_token, start_result],
            )

        with gr.Tab("3. Monitor result"):
            refresh_btn = gr.Button("Refresh job")
            status_json = gr.Textbox(label="Current job status", lines=18)
            refresh_btn.click(refresh_job, inputs=[base_url, job_id, job_token], outputs=[status_json])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, inline=True, debug=True, server_name='0.0.0.0', server_port=7860)
