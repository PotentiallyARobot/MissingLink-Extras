import json
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path

import requests
from twilio.rest import Client

try:
    from google.colab import userdata
except Exception:
    userdata = None

ROOT = Path(__file__).resolve().parent
SERVERLESS_DIR = ROOT / "twilio-serverless"
DEPLOY_INFO_PATH = ROOT / "deployment.json"
ENV_PATH = SERVERLESS_DIR / ".env.generated"


class DeployError(RuntimeError):
    pass


def _get_secret(name: str, default: str | None = None) -> str | None:
    if name in os.environ:
        return os.environ[name]
    if userdata is not None:
        try:
            value = userdata.get(name)
            if value:
                return value
        except Exception:
            pass
    return default


def _ensure_node():
    if shutil.which("node") and shutil.which("npm"):
        return
    subprocess.run(["bash", "-lc", "apt-get update -y && apt-get install -y nodejs npm"], check=True)


def _ensure_sync_service(client: Client, friendly_name: str = "PhoneAgentStatus") -> str:
    services = client.sync.v1.services.list(limit=50)
    for service in services:
        if getattr(service, "friendly_name", None) == friendly_name:
            return service.sid
    service = client.sync.v1.services.create(friendly_name=friendly_name)
    return service.sid


def _write_env(values: dict[str, str]):
    lines = [f"{k}={v}" for k, v in values.items()]
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_base_url(text: str) -> str | None:
    matches = re.findall(r"https://[a-zA-Z0-9\-]+\.twil\.io", text)
    return matches[-1] if matches else None


def deploy(environment: str = "dev") -> dict:
    account_sid = _get_secret("TWILIO_ACCOUNT_SID")
    auth_token = _get_secret("TWILIO_AUTH_TOKEN")
    from_number = _get_secret("TWILIO_FROM_NUMBER")
    openai_api_key = _get_secret("OPENAI_API_KEY")

    missing = [
        name
        for name, value in {
            "TWILIO_ACCOUNT_SID": account_sid,
            "TWILIO_AUTH_TOKEN": auth_token,
            "TWILIO_FROM_NUMBER": from_number,
            "OPENAI_API_KEY": openai_api_key,
        }.items()
        if not value
    ]
    if missing:
        raise DeployError(f"Missing required secrets: {', '.join(missing)}")

    client = Client(account_sid, auth_token)
    sync_service_sid = _ensure_sync_service(client)
    admin_key = _get_secret("PHONE_AGENT_ADMIN_KEY") or secrets.token_urlsafe(24)

    env_values = {
        "OPENAI_API_KEY": openai_api_key,
        "OPENAI_MODEL": _get_secret("OPENAI_MODEL", "gpt-5.4-nano"),
        "TWILIO_FROM_NUMBER": from_number,
        "SYNC_SERVICE_SID": sync_service_sid,
        "APP_ADMIN_KEY": admin_key,
        "DEFAULT_MAX_TURNS": _get_secret("DEFAULT_MAX_TURNS", "6"),
        "DEFAULT_LANGUAGE": _get_secret("DEFAULT_LANGUAGE", "en-US"),
    }
    _write_env(env_values)

    _ensure_node()

    env = os.environ.copy()
    env["TWILIO_ACCOUNT_SID"] = account_sid
    env["TWILIO_AUTH_TOKEN"] = auth_token
    env["TWILIO_SERVERLESS_API_CONCURRENCY"] = "1"

    subprocess.run(["npm", "install"], cwd=SERVERLESS_DIR, check=True, env=env)

    cmd = [
        "npx",
        "twilio-run",
        "deploy",
        "--env",
        str(ENV_PATH),
        "--environment",
        environment,
    ]
    proc = subprocess.run(
        cmd,
        cwd=SERVERLESS_DIR,
        env=env,
        capture_output=True,
        text=True,
    )

    combined_output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise DeployError(combined_output.strip())

    base_url = _parse_base_url(combined_output)
    if not base_url:
        raise DeployError(
            "Deployment succeeded but I could not parse the Twilio Functions base URL from the output.\n"
            + combined_output.strip()
        )

    deploy_info = {
        "base_url": base_url,
        "admin_key": admin_key,
        "sync_service_sid": sync_service_sid,
        "environment": environment,
    }
    DEPLOY_INFO_PATH.write_text(json.dumps(deploy_info, indent=2), encoding="utf-8")
    return deploy_info


if __name__ == "__main__":
    info = deploy()
    print(json.dumps(info, indent=2))
