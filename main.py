import os
import uuid
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, Response
from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.genai.types import Part, Content, Blob
from agent import root_agent
from pydub import AudioSegment

app = FastAPI()
CLOUD_RUN_URL = "wss://bidi-streamvoice-836864412652.us-central1.run.app"
APP_NAME = "ADK Gemini Voice Agent"

# --- Twilio audio conversion --- #
def convert_mulaw_to_pcm(raw_bytes):
    audio = AudioSegment(
        data=raw_bytes,
        sample_width=1,
        frame_rate=8000,
        channels=1
    )
    converted = audio.set_frame_rate(16000).set_sample_width(2)
    return converted.raw_data

# --- ADK session starter --- #
async def start_agent_session(user_id: str, is_audio=True):
    runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
    run_config = RunConfig(response_modalities=["AUDIO" if is_audio else "TEXT"])
    queue = LiveRequestQueue()
    live = runner.run_live(session=session, live_request_queue=queue, run_config=run_config)
    return live, queue

# --- Twilio-compatible webhook --- #
@app.post("/voice")
async def voice_webhook():
    session_id = str(uuid.uuid4())
    stream_url = f"{CLOUD_RUN_URL}/twilio/{session_id}"
    xml = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<Response>
  <Connect>
    <Stream url=\"{stream_url}\" />
  </Connect>
  <Say>Welcome to Quantum Veda AI Assistant.</Say>
</Response>"""
    return Response(content=xml.strip(), media_type="application/xml")

# --- Twilio WebSocket handler --- #
@app.websocket("/twilio/{user_id}")
async def twilio_ws(websocket: WebSocket, user_id: str):
    await websocket.accept()
    print(f"✅ Twilio WebSocket connected: {user_id}")
    live_events, queue = await start_agent_session(user_id, is_audio=True)

    async def from_twilio():
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("event") == "media":
                mulaw_audio = base64.b64decode(data["media"]["payload"])
                pcm_audio = convert_mulaw_to_pcm(mulaw_audio)
                queue.send_realtime(Blob(data=pcm_audio, mime_type="audio/pcm"))

    async def to_twilio():
        async for event in live_events:
            part = event.content and event.content.parts and event.content.parts[0]
            if part and part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                payload = base64.b64encode(part.inline_data.data).decode("ascii")
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "media": {"payload": payload}
                }))

    await asyncio.wait([
        asyncio.create_task(from_twilio()),
        asyncio.create_task(to_twilio())
    ], return_when=asyncio.FIRST_EXCEPTION)
    queue.close()
    print(f"❌ Twilio disconnected: {user_id}")

# --- Basic health check --- #
@app.get("/")
async def root():
    return {"status": "running"}
