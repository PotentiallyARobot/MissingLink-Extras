"""
AI Browser Agent — Vision-guided browser automation with real cursor control.

Uses a real browser (Chrome via subprocess), takes screenshots, sends them to
OpenAI GPT-4o vision, and executes physical mouse/keyboard actions via PyAutoGUI.

Requirements:
    pip install openai pyautogui pillow

Usage:
    python ai_browser_agent.py \
        --url "https://mail.google.com" \
        --task "Log into my email, find the latest email from Amazon, and tell me what it says" \
        --email "you@gmail.com" \
        --password "yourpassword"

    # Or interactive mode:
    python ai_browser_agent.py
"""

import argparse
import base64
import io
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

try:
    import pyautogui
except ImportError:
    sys.exit("Missing pyautogui. Install with:  pip install pyautogui")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing openai. Install with:  pip install openai")

try:
    from PIL import Image
except ImportError:
    sys.exit("Missing Pillow. Install with:  pip install pillow")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Safety: PyAutoGUI will raise an exception if the mouse hits a screen corner.
# This lets you abort by flinging the mouse to a corner.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.15  # small pause between pyautogui calls

MAX_STEPS = 40          # hard cap so it can't loop forever
SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """\
You are an AI browser-automation agent. You see a screenshot of a desktop with
a web browser open. Your job is to accomplish the user's task by issuing ONE
action per turn.

Available actions (respond with EXACTLY ONE JSON object, no markdown fences):

1. CLICK at coordinates:
   {"action": "click", "x": 450, "y": 320, "description": "Click the Sign-in button"}

2. DOUBLE CLICK:
   {"action": "double_click", "x": 450, "y": 320, "description": "Double-click the file"}

3. RIGHT CLICK:
   {"action": "right_click", "x": 450, "y": 320, "description": "Right-click for context menu"}

4. TYPE text (typed into whatever is currently focused):
   {"action": "type", "text": "hello@example.com", "description": "Type email address"}

5. PRESS a key or combo (e.g. enter, tab, ctrl+a, backspace, escape):
   {"action": "press", "key": "enter", "description": "Submit the form"}

6. SCROLL (positive = down, negative = up):
   {"action": "scroll", "amount": -3, "x": 700, "y": 400, "description": "Scroll up to see header"}

7. WAIT (seconds) for page to load:
   {"action": "wait", "seconds": 3, "description": "Wait for page to load"}

8. DONE — the task is complete:
   {"action": "done", "summary": "Successfully logged in and found the email. It says ..."}

9. FAIL — the task cannot be completed:
   {"action": "fail", "reason": "CAPTCHA requires human intervention"}

Guidelines:
- x,y are PIXEL coordinates on the screenshot (top-left is 0,0).
- Before clicking an input field, CLICK it first, then TYPE in the next step.
- If you need to clear a field first, click it, then press ctrl+a, then type.
- Always describe what you see and why you chose that action.
- If a page is loading, use WAIT.
- Never guess — if you can't see the target element, scroll or wait.
- Coordinates must be integers within the screenshot dimensions.
- For sensitive fields (passwords), type carefully — you'll see bullets/dots.
- When the task is fully done, use the DONE action with a helpful summary.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def take_screenshot() -> Image.Image:
    """Capture the entire screen and return a PIL Image."""
    img = pyautogui.screenshot()
    return img


def screenshot_to_base64(img: Image.Image, max_width: int = 1920) -> str:
    """Resize if needed and convert to base64 JPEG for the API."""
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def save_screenshot(img: Image.Image, step: int):
    """Save screenshot to disk for debugging."""
    path = SCREENSHOT_DIR / f"step_{step:03d}.png"
    img.save(path)
    return path


def human_move(x: int, y: int, duration_range=(0.3, 0.8)):
    """
    Move the mouse to (x, y) with a human-like curve and random duration.
    PyAutoGUI's moveTo with a tween gives a smooth, non-linear path.
    """
    duration = random.uniform(*duration_range)
    # Add slight randomness to the target so it's not pixel-perfect every time
    jitter_x = x + random.randint(-2, 2)
    jitter_y = y + random.randint(-2, 2)
    pyautogui.moveTo(jitter_x, jitter_y, duration=duration, tween=pyautogui.easeOutQuad)


def human_type(text: str, interval_range=(0.04, 0.12)):
    """Type text with randomized per-character delays to mimic a real human."""
    for char in text:
        pyautogui.typewrite(char, interval=random.uniform(*interval_range))
        # Occasionally add a micro-pause (like a human hesitating)
        if random.random() < 0.05:
            time.sleep(random.uniform(0.15, 0.4))


def open_browser(url: str):
    """Open Chrome (or default browser) at the given URL."""
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-a", "Google Chrome", url])
        elif system == "Windows":
            subprocess.Popen(["start", "chrome", url], shell=True)
        else:
            # Linux — try chrome, chromium, then xdg-open
            for browser in ["google-chrome", "chromium-browser", "chromium", "xdg-open"]:
                if subprocess.run(["which", browser], capture_output=True).returncode == 0:
                    subprocess.Popen([browser, url])
                    break
    except Exception as e:
        print(f"[!] Could not auto-open browser: {e}")
        print(f"    Please open {url} manually.")

    print(f"[*] Waiting for browser to open {url}...")
    time.sleep(4)


# ---------------------------------------------------------------------------
# Agent core
# ---------------------------------------------------------------------------

def call_vision_model(client: OpenAI, messages: list, model: str) -> dict:
    """Send the conversation to GPT and parse the JSON action response."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps them anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find a JSON object in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"Could not parse action JSON from model response:\n{raw}")


def execute_action(action: dict) -> str:
    """Execute a single action dict and return a status message."""
    act = action.get("action", "").lower()
    desc = action.get("description", "")

    if act == "click":
        x, y = int(action["x"]), int(action["y"])
        print(f"  → CLICK ({x}, {y}): {desc}")
        human_move(x, y)
        time.sleep(random.uniform(0.05, 0.15))
        pyautogui.click()
        return f"Clicked at ({x}, {y})"

    elif act == "double_click":
        x, y = int(action["x"]), int(action["y"])
        print(f"  → DOUBLE CLICK ({x}, {y}): {desc}")
        human_move(x, y)
        pyautogui.doubleClick()
        return f"Double-clicked at ({x}, {y})"

    elif act == "right_click":
        x, y = int(action["x"]), int(action["y"])
        print(f"  → RIGHT CLICK ({x}, {y}): {desc}")
        human_move(x, y)
        pyautogui.rightClick()
        return f"Right-clicked at ({x}, {y})"

    elif act == "type":
        text = action["text"]
        masked = text if len(text) < 4 else text[:2] + "*" * (len(text) - 2)
        print(f"  → TYPE: {masked} — {desc}")
        human_type(text)
        return f"Typed {len(text)} characters"

    elif act == "press":
        key = action["key"]
        print(f"  → PRESS: {key} — {desc}")
        if "+" in key:
            # Combo like ctrl+a
            parts = [k.strip() for k in key.split("+")]
            pyautogui.hotkey(*parts)
        else:
            pyautogui.press(key)
        return f"Pressed {key}"

    elif act == "scroll":
        amount = int(action["amount"])
        x = int(action.get("x", pyautogui.position()[0]))
        y = int(action.get("y", pyautogui.position()[1]))
        print(f"  → SCROLL {amount} at ({x}, {y}): {desc}")
        human_move(x, y)
        pyautogui.scroll(amount)
        return f"Scrolled {amount} at ({x}, {y})"

    elif act == "wait":
        secs = min(float(action.get("seconds", 2)), 15)  # cap at 15s
        print(f"  → WAIT {secs}s: {desc}")
        time.sleep(secs)
        return f"Waited {secs}s"

    elif act == "done":
        summary = action.get("summary", "Task completed.")
        return f"DONE: {summary}"

    elif act == "fail":
        reason = action.get("reason", "Unknown reason.")
        return f"FAIL: {reason}"

    else:
        return f"Unknown action: {act}"


def run_agent(
    url: str,
    task: str,
    email: str = "",
    password: str = "",
    api_key: str = "",
    model: str = "gpt-4o",
    verbose: bool = True,
):
    """Main agent loop."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # Build the initial user message with credentials context
    credential_note = ""
    if email:
        credential_note += f"\nEmail/username to use: {email}"
    if password:
        credential_note += f"\nPassword to use: {password}"

    task_description = f"""TASK: {task}
URL: {url}{credential_note}

I have opened the browser to the URL. Look at the screenshot and begin."""

    # Open the browser
    open_browser(url)

    # Conversation history for the model
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print(f"\n{'='*60}")
    print(f"  AI Browser Agent")
    print(f"  URL:  {url}")
    print(f"  Task: {task}")
    print(f"  Model: {model}")
    print(f"{'='*60}\n")
    print("[*] Starting automation loop (move mouse to any corner to abort)\n")

    for step in range(1, MAX_STEPS + 1):
        print(f"--- Step {step}/{MAX_STEPS} ---")

        # 1. Screenshot
        time.sleep(1)  # brief pause for any animations/loads
        img = take_screenshot()
        saved = save_screenshot(img, step)
        b64 = screenshot_to_base64(img)
        if verbose:
            print(f"  Screenshot saved: {saved}")

        # 2. Build the message for this turn
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
            },
        ]
        if step == 1:
            user_content.append({"type": "text", "text": task_description})
        else:
            user_content.append({
                "type": "text",
                "text": f"Here is the updated screenshot after the last action. Continue with the task. Step {step}/{MAX_STEPS}.",
            })

        messages.append({"role": "user", "content": user_content})

        # 3. Ask the vision model
        try:
            action = call_vision_model(client, messages, model)
        except Exception as e:
            print(f"  [!] Model error: {e}")
            print("  Retrying in 3s...")
            time.sleep(3)
            continue

        # Add assistant response to history (as text for context)
        messages.append({"role": "assistant", "content": json.dumps(action)})

        # 4. Execute the action
        result = execute_action(action)
        print(f"  Result: {result}\n")

        # 5. Check terminal states
        if action.get("action") == "done":
            print(f"\n{'='*60}")
            print(f"  ✅ TASK COMPLETE")
            print(f"  {action.get('summary', '')}")
            print(f"{'='*60}\n")
            return action.get("summary", "Done.")

        if action.get("action") == "fail":
            print(f"\n{'='*60}")
            print(f"  ❌ TASK FAILED")
            print(f"  {action.get('reason', '')}")
            print(f"{'='*60}\n")
            return None

        # Keep conversation history manageable — keep last 10 turns
        # (each turn = 2 messages: user screenshot + assistant action)
        if len(messages) > 22:  # system + 10 turns × 2
            messages = [messages[0]] + messages[-20:]

    print(f"\n[!] Reached max steps ({MAX_STEPS}). Stopping.")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Browser Agent — GPT Vision-guided browser automation with real cursor control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Login and read email:
  python ai_browser_agent.py \\
      --url "https://mail.google.com" \\
      --task "Log in, find the latest email from Amazon, summarize it" \\
      --email "me@gmail.com" --password "hunter2"

  # Fill out a form:
  python ai_browser_agent.py \\
      --url "https://example.com/form" \\
      --task "Fill out the contact form with name John Doe, email john@test.com, message Hello"

  # Interactive mode (prompts for everything):
  python ai_browser_agent.py
        """,
    )
    parser.add_argument("--url", type=str, help="Starting URL")
    parser.add_argument("--task", type=str, help="What the agent should accomplish")
    parser.add_argument("--email", type=str, default="", help="Email/username if login required")
    parser.add_argument("--password", type=str, default="", help="Password if login required")
    parser.add_argument("--api-key", type=str, default="", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Vision model to use (default: gpt-4o)")
    parser.add_argument("--max-steps", type=int, default=40, help="Max automation steps (default: 40)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.max_steps:
        global MAX_STEPS
        MAX_STEPS = args.max_steps

    # Interactive mode if no args
    url = args.url or input("Enter the URL: ").strip()
    task = args.task or input("What should I do? ").strip()
    email = args.email or input("Email/username (blank to skip): ").strip()
    password = args.password
    if not password and email:
        import getpass
        password = getpass.getpass("Password (blank to skip): ")

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        api_key = input("OpenAI API key: ").strip()

    run_agent(
        url=url,
        task=task,
        email=email,
        password=password,
        api_key=api_key,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
