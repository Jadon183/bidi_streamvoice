import os
import json
import base64
import asyncio
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from google.genai.types import Part, Content, Blob
from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig

from agent import root_agent

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()
STREAM_URL = os.getenv("TWILIO_STREAM_URL")

APP_NAME = "ADK Streaming example"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions = {}

# ---------- Gemini Agent Setup ----------

async def start_agent_session(user_id: str, is_audio: bool = True):
    runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)
    session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
    modality = "AUDIO" if is_audio else "TEXT"
    run_config = RunConfig(response_modalities=[modality])
    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live(session=session, live_request_queue=live_request_queue, run_config=run_config)
    return live_events, live_request_queue

# ---------- Twilio Call Entry ----------

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def handle_outbound_call():
    response = VoiceResponse()
    connect = Connect()
    connect.stream(
        url="wss://{STREAM_URL}/twilio/media",  # Replace with your deployed WebSocket
        track="both_tracks",
        content_type="audio/l16;rate=16000"
    )
    response.append(connect)
    return str(response)

# ---------- Twilio Media WebSocket ----------

@app.websocket("/twilio/media")
async def twilio_media_stream(websocket: WebSocket):
    await websocket.accept()
    user_id = f"user_{id(websocket)}"
    print(f"🔌 Connected: {user_id}")

    live_events, live_request_queue = await start_agent_session(user_id, is_audio=True)
    active_sessions[user_id] = live_request_queue

    # Task to send audio from Gemini -> Twilio
    async def send_agent_audio():
        try:
            async for event in live_events:
                part = event.content.parts[0] if event.content and event.content.parts else None
                if part and part.inline_data and part.inline_data.mime_type == "audio/pcm":
                    audio_bytes = part.inline_data.data
                    base64_audio = base64.b64encode(audio_bytes).decode("ascii")

                    twilio_msg = {
                        "event": "media",
                        "media": {"payload": base64_audio}
                    }
                    await websocket.send_json(twilio_msg)
        except Exception as e:
            print(f"⚠️ Agent audio send error: {e}")

    send_task = asyncio.create_task(send_agent_audio())

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("event")

            if event_type == "start":
                print(f"📞 Call started")

            elif event_type == "media":
                payload = data["media"]["payload"]
                audio_bytes = base64.b64decode(payload)
                blob = Blob(data=audio_bytes, mime_type="audio/pcm")
                live_request_queue.send_realtime(blob)

            elif event_type == "stop":
                print(f"📴 Call ended")
                break

    except WebSocketDisconnect:
        print(f"❌ Disconnected: {user_id}")
    except Exception as e:
        print(f"❗ Error: {e}")
    finally:
        send_task.cancel()
        live_request_queue.close()
        active_sessions.pop(user_id, None)

# ---------- Optional Browser SSE ----------

async def agent_to_client_sse(live_events):
    async for event in live_events:
        if event.turn_complete or event.interrupted:
            yield f"data: {json.dumps({'done': True})}\n\n"
            continue

        part = event.content and event.content.parts and event.content.parts[0]
        if not part:
            continue

        if part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
            message = {
                "mime_type": "audio/pcm",
                "data": base64.b64encode(part.inline_data.data).decode("ascii")
            }
            yield f"data: {json.dumps(message)}\n\n"
        elif part.text and event.partial:
            yield f"data: {json.dumps({'mime_type': 'text/plain', 'data': part.text})}\n\n"

@app.get("/")
def root():
    return {"message": "FastAPI ADK Server Running"}

@app.get("/events/{user_id}")
async def sse_endpoint(user_id: int, is_audio: str = "false"):
    user_id_str = str(user_id)
    live_events, live_request_queue = await start_agent_session(user_id_str, is_audio == "true")
    active_sessions[user_id_str] = live_request_queue

    def cleanup():
        live_request_queue.close()
        active_sessions.pop(user_id_str, None)

    async def event_generator():
        try:
            async for data in agent_to_client_sse(live_events):
                yield data
        except Exception as e:
            print(f"[SSE Error]: {e}")
        finally:
            cleanup()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/send/{user_id}")
async def send_message_endpoint(user_id: int, request: Request):
    user_id_str = str(user_id)
    live_request_queue = active_sessions.get(user_id_str)
    if not live_request_queue:
        return {"error": "Session not found"}

    message = await request.json()
    mime_type = message["mime_type"]
    data = message["data"]

    if mime_type == "text/plain":
        content = Content(role="user", parts=[Part.from_text(text=data)])
        live_request_queue.send_content(content=content)
    elif mime_type == "audio/pcm":
        decoded_data = base64.b64decode(data)
        live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
    else:
        return {"error": f"Unsupported mime type: {mime_type}"}

    return {"status": "sent"}
