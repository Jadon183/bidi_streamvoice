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

app = FastAPI()
CLOUD_RUN_URL = "https://bidi-streamvoice-836864412652.us-central1.run.app"
APP_NAME = "ADK Streaming Voice"

# --- Shared Agent Starter --- #
async def start_agent_session(user_id: str, is_audio=True):
    runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
    modality = "AUDIO" if is_audio else "TEXT"
    config = RunConfig(response_modalities=[modality])
    queue = LiveRequestQueue()
    live_events = runner.run_live(session=session, live_request_queue=queue, run_config=config)
    return live_events, queue

# --- Twilio-Compatible TwiML Webhook --- #
@app.post("/voice")
async def voice_webhook():
    session_id = str(uuid.uuid4())
    stream_url = f"{CLOUD_RUN_URL}/twilio/{session_id}"
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{stream_url}" />
  </Connect>
  <Say>Welcome to Quantum Veda AI Assistant.</Say>
</Response>"""
    return Response(content=xml.strip(), media_type="application/xml")

# --- Twilio Media Stream WebSocket Handler --- #
@app.websocket("/twilio/{user_id}")
async def twilio_ws_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    print(f"[Twilio WS] Connected: {user_id}")
    live_events, queue = await start_agent_session(user_id, is_audio=True)

    async def receive_from_twilio():
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if data.get("event") == "media":
                audio = base64.b64decode(data["media"]["payload"])
                queue.send_realtime(Blob(data=audio, mime_type="audio/pcm"))

    async def send_to_twilio():
        async for event in live_events:
            part = event.content and event.content.parts and event.content.parts[0]
            if part and part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                payload = base64.b64encode(part.inline_data.data).decode("ascii")
                reply = {"event": "media", "media": {"payload": payload}}
                await websocket.send_text(json.dumps(reply))

    await asyncio.wait([
        asyncio.create_task(receive_from_twilio()),
        asyncio.create_task(send_to_twilio())
    ], return_when=asyncio.FIRST_EXCEPTION)

    queue.close()
    print(f"[Twilio WS] Disconnected: {user_id}")

# --- Browser/Custom Agent WebSocket (your original format) --- #
@app.websocket("/ws/{user_id}")
async def browser_ws_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    print(f"[Browser WS] Connected: {user_id}")
    live_events, queue = await start_agent_session(user_id, is_audio=True)

    async def agent_to_client():
        async for event in live_events:
            part = event.content and event.content.parts and event.content.parts[0]
            if not part:
                continue
            if part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
                payload = base64.b64encode(part.inline_data.data).decode("ascii")
                await websocket.send_text(json.dumps({"mime_type": "audio/pcm", "data": payload}))
            elif part.text:
                await websocket.send_text(json.dumps({"mime_type": "text/plain", "data": part.text}))

    async def client_to_agent():
        while True:
            msg = json.loads(await websocket.receive_text())
            if msg["mime_type"] == "text/plain":
                queue.send_content(Content(role="user", parts=[Part.from_text(text=msg["data"])]))
            elif msg["mime_type"] == "audio/pcm":
                queue.send_realtime(Blob(data=base64.b64decode(msg["data"]), mime_type="audio/pcm"))

    await asyncio.wait([
        asyncio.create_task(agent_to_client()),
        asyncio.create_task(client_to_agent())
    ], return_when=asyncio.FIRST_EXCEPTION)

    queue.close()
    print(f"[Browser WS] Disconnected: {user_id}")

# --- Simple Health Check --- #
@app.get("/")
async def root():
    return {"status": "running"}
