# Colab Phone Agent (Twilio + OpenAI)

This project gives you a **Colab control panel** plus a **Twilio Functions backend** so you can place an outbound phone call, pursue a simple objective, hang up when done, and then read the extracted result back in the UI.

## What this package does

- Starts a simple UI in Colab for entering:
  - the destination phone number
  - the call objective
  - an optional contact name
- Deploys the public webhook backend to **Twilio Functions** (so you do **not** need ngrok)
- Uses Twilio `<Gather input="speech dtmf">` to listen for spoken replies or keypad input during the call
- Uses **OpenAI Responses API** with `gpt-5.4-nano` as the text decision-maker
- Stores call state and the final extracted answer in **Twilio Sync** so the Colab UI can poll and display the result

## Important limits

- This build uses **Twilio TTS (`<Say>`)** for speaking on the call. That is deliberate: it is the simplest no-ngrok / no-websocket architecture. OpenAI Realtime is better for lower latency voice bots, but it requires a live public websocket endpoint rather than a plain Twilio Function.
- On a normal PSTN phone call, the only practical way to “press a button” remotely is **DTMF**. This project can send DTMF digits into the call when the agent decides to do so, but there is no separate physical-button channel over a standard phone call.
- Trial Twilio accounts can usually only call **verified** numbers.
- For trust and legality, the agent prompt is set up to **identify itself as an automated assistant** at the start of the call.

## Project layout

- `launch_colab.py` — starts the local Gradio UI in Colab
- `deploy_twilio_from_colab.py` — prepares `.env`, ensures a Sync service exists, installs dependencies, and deploys Twilio Functions
- `requirements-colab.txt` — Python dependencies for the notebook side
- `twilio-serverless/` — Twilio Functions project that handles the actual call logic

## Quick start in Colab

1. Upload and unzip this project in Colab.
2. Put these secrets in **Colab userdata**:
   - `TWILIO_ACCOUNT_SID`
   - `TWILIO_AUTH_TOKEN`
   - `TWILIO_FROM_NUMBER`
   - `OPENAI_API_KEY`
3. Install Python deps:

```python
!pip -q install -r requirements-colab.txt
```

4. Start the UI:

```python
!python launch_colab.py
```

5. In the UI:
   - click **Deploy / refresh backend** once
   - wait for the Twilio Functions deployment URL
   - enter the number and the objective
   - click **Start call**
   - click **Refresh job** to poll the result

## Example objective

```text
Call Tyler and find out where he put his coat. When you have the information, say thank you, goodbye, and hang up.
```

## Notes on deployment

The deployment step uses the Twilio Serverless Toolkit / `twilio-run` style workflow. If the automatic deploy step fails in your notebook environment, you can still deploy manually from the `twilio-serverless/` folder after inspecting the generated `.env.generated` file.
