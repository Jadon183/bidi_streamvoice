import os
import json
import asyncio
import base64
import uuid
import warnings

from pathlib import Path
from dotenv import load_dotenv

from google.genai.types import (
    Part,
    Content,
    Blob,
)

from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from agent import root_agent



#
# ADK Streaming
#

# Load Gemini API Key

from fastapi import WebSocket
#from twilio_adapter import twilio_audio_stream_handler, adk_response_to_twilio
import asyncio
import base64
import json

from google.genai.types import Content, Part, Blob
from google.adk.agents import LiveRequestQueue
from starlette.websockets import WebSocket

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
AUDIO_MIME = "audio/pcm"

async def twilio_audio_stream_handler(websocket: WebSocket, live_request_queue: LiveRequestQueue):
    while True:
        try:
            # Raw PCM from Twilio
            frame = await websocket.receive_bytes()
            # Directly send to ADK agent as Blob
            live_request_queue.send_realtime(Blob(data=frame, mime_type=AUDIO_MIME))
        except Exception as e:
            print("[Twilio Stream Error]", e)
            break

async def adk_response_to_twilio(websocket: WebSocket, live_events):
    async for event in live_events:
        part: Part = (
            event.content and event.content.parts and event.content.parts[0]
        )
        if part and part.inline_data and part.inline_data.mime_type.startswith("audio/pcm"):
            audio_data = part.inline_data.data
            await websocket.send_bytes(audio_data)
            print(f"[ADK → Twilio] Sent {len(audio_data)} bytes")

APP_NAME = "ADK Streaming example"


async def start_agent_session(user_id, is_audio=False):
    """Starts an agent session"""

    # Create a Runner
    runner = InMemoryRunner(
        app_name=APP_NAME,
        agent=root_agent,
    )

    # Create a Session
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,  # Replace with actual user ID
    )

    # Set response modality
    modality = "AUDIO" if is_audio else "TEXT"
    run_config = RunConfig(response_modalities=[modality])

    # Create a LiveRequestQueue for this session
    live_request_queue = LiveRequestQueue()

    # Start agent session
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config,
    )
    return live_events, live_request_queue


async def agent_to_client_messaging(websocket, live_events):
    """Agent to client communication"""
    while True:
        async for event in live_events:

            # If the turn complete or interrupted, send it
            if event.turn_complete or event.interrupted:
                message = {
                    "turn_complete": event.turn_complete,
                    "interrupted": event.interrupted,
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT TO CLIENT]: {message}")
                continue

            # Read the Content and its first Part
            part: Part = (
                event.content and event.content.parts and event.content.parts[0]
            )
            if not part:
                continue

            # If it's audio, send Base64 encoded audio data
            is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/pcm")
            if is_audio:
                audio_data = part.inline_data and part.inline_data.data
                if audio_data:
                    message = {
                        "mime_type": "audio/pcm",
                        "data": base64.b64encode(audio_data).decode("ascii")
                    }
                    await websocket.send_text(json.dumps(message))
                    print(f"[AGENT TO CLIENT]: audio/pcm: {len(audio_data)} bytes.")
                    continue

            # If it's text and a parial text, send it
            if part.text and event.partial:
                message = {
                    "mime_type": "text/plain",
                    "data": part.text
                }
                await websocket.send_text(json.dumps(message))
                print(f"[AGENT TO CLIENT]: text/plain: {message}")


async def client_to_agent_messaging(websocket, live_request_queue):
    """Client to agent communication"""
    while True:
        # Decode JSON message
        message_json = await websocket.receive_text()
        message = json.loads(message_json)
        mime_type = message["mime_type"]
        data = message["data"]

        # Send the message to the agent
        if mime_type == "text/plain":
            # Send a text message
            content = Content(role="user", parts=[Part.from_text(text=data)])
            live_request_queue.send_content(content=content)
            print(f"[CLIENT TO AGENT]: {data}")
        elif mime_type == "audio/pcm":
            # Send an audio data
            decoded_data = base64.b64decode(data)
            live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
        else:
            raise ValueError(f"Mime type not supported: {mime_type}")


#
# FastAPI web app
#

app = FastAPI()



@app.websocket("/twilio/{user_id}")
async def twilio_ws_endpoint(websocket: WebSocket, user_id: str):
    """Direct WebSocket endpoint for Twilio PCM audio"""
    await websocket.accept()
    print(f"[Twilio] Connected for user {user_id}")

    # Start ADK session
    user_id_str = str(user_id)
    live_events, live_request_queue = await start_agent_session(user_id_str, is_audio=True)

    # Two-way streaming: Twilio → ADK and ADK → Twilio
    twilio_to_adk_task = asyncio.create_task(
        twilio_audio_stream_handler(websocket, live_request_queue)
    )
    adk_to_twilio_task = asyncio.create_task(
        adk_response_to_twilio(websocket, live_events)
    )

    await asyncio.wait(
        [twilio_to_adk_task, adk_to_twilio_task],
        return_when=asyncio.FIRST_EXCEPTION
    )

    live_request_queue.close()
    print(f"[Twilio] Disconnected user {user_id}")


@app.get("/")
async def root():
    """Serves the index.html"""
    return "Sab changa hai"

from fastapi.responses import Response

CLOUD_RUN_URL = "https://bidi-streamvoice-836864412652.us-central1.run.app"

@app.post("/voice")
async def voice_webhook():
    session_id = str(uuid.uuid4())  # Optional: generate random ID for each call
    stream_url = f"{CLOUD_RUN_URL}/twilio/{session_id}"

    print(f"[Twilio Webhook] Starting stream to {stream_url}")
    return Response(content=f"""
        <Response>
            <Start>
                <Stream url="{stream_url}"/>
            </Start>
            <Say>Welcome to Quantum Veda AI Assistant. How can I help you today?</Say>
        </Response>
    """, media_type="application/xml")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int, is_audio: str):
    """Client websocket endpoint"""

    # Wait for client connection
    await websocket.accept()
    print(f"Client #{user_id} connected, audio mode: {is_audio}")

    # Start agent session
    user_id_str = str(user_id)
    live_events, live_request_queue = await start_agent_session(user_id_str, is_audio == "true")

    # Start tasks
    agent_to_client_task = asyncio.create_task(
        agent_to_client_messaging(websocket, live_events)
    )
    client_to_agent_task = asyncio.create_task(
        client_to_agent_messaging(websocket, live_request_queue)
    )

    # Wait until the websocket is disconnected or an error occurs
    tasks = [agent_to_client_task, client_to_agent_task]
    await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # Close LiveRequestQueue
    live_request_queue.close()

    # Disconnected
    print(f"Client #{user_id} disconnected")
