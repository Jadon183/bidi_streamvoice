import os
import json
import base64
import warnings

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Gather

from google.genai.types import Part, Content, Blob
from google.adk.runners import InMemoryRunner
from google.adk.agents import LiveRequestQueue
from google.adk.agents.run_config import RunConfig

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from agent import root_agent

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

APP_NAME = "ADK Streaming example"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure per environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions = {}

# ---------- Gemini Agent Setup ----------

async def start_agent_session(user_id, is_audio=False):
    runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)
    session = await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id
    )
    modality = "AUDIO" if is_audio else "TEXT"
    run_config = RunConfig(response_modalities=[modality])
    live_request_queue = LiveRequestQueue()
    live_events = runner.run_live(
        session=session,
        live_request_queue=live_request_queue,
        run_config=run_config
    )
    return live_events, live_request_queue

async def get_agent_response(user_id: str, message: str) -> str:
    if user_id not in active_sessions:
        live_events, live_request_queue = await start_agent_session(user_id)
        active_sessions[user_id] = live_request_queue
    else:
        live_request_queue = active_sessions[user_id]
        live_events, _ = await start_agent_session(user_id)

    content = Content(role="user", parts=[Part.from_text(text=message)])
    live_request_queue.send_content(content=content)

    async for event in live_events:
        part = event.content.parts[0] if event.content and event.content.parts else None
        if part and part.text:
            return part.text
    return "Sorry, I couldn't find an answer."

# ---------- Twilio Multi-turn Voice Webhook ----------

@app.post("/twilio/voice", response_class=PlainTextResponse)
async def handle_voice_call():
    """Initial voice greeting"""
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action="/twilio/handle_speech",
        method="POST",
        timeout=5,
        language="en-US"
    )
    gather.say("Hello! You are connected to the AI assistant. Ask your question after the beep.")
    response.append(gather)
    response.say("I didn’t hear anything. Goodbye!")
    response.hangup()
    return str(response)

@app.post("/twilio/handle_speech", response_class=PlainTextResponse)
async def handle_speech(request: Request):
    """Handle STT input and respond using Gemini, then loop"""
    form = await request.form()
    speech_text = form.get("SpeechResult", "").strip().lower()
    caller = form.get("From")

    print(f"[CALLER {caller}]: {speech_text}")

    response = VoiceResponse()

    # Exit conditions
    if speech_text in ("exit", "quit", "bye", "goodbye", "stop"):
        response.say("Goodbye! It was nice talking to you.")
        response.hangup()
        return str(response)

    if not speech_text:
        response.say("Sorry, I didn't catch that.")
        gather = Gather(input="speech", action="/twilio/handle_speech", method="POST", timeout=5, language="en-US")
        gather.say("Please ask your question again.")
        response.append(gather)
        return str(response)

    gemini_response = await get_agent_response(caller, speech_text)

    response.say(gemini_response, voice="Polly.Joanna", language="en-US")
    gather = Gather(input="speech", action="/twilio/handle_speech", method="POST", timeout=5, language="en-US")
    gather.say("You can ask another question or say goodbye to end the call.")
    response.append(gather)

    return str(response)

# ---------- Optional: Browser SSE Streaming ----------

async def agent_to_client_sse(live_events):
    async for event in live_events:
        if event.turn_complete or event.interrupted:
            message = {
                "turn_complete": event.turn_complete,
                "interrupted": event.interrupted,
            }
            yield f"data: {json.dumps(message)}\n\n"
            continue

        part = event.content and event.content.parts and event.content.parts[0]
        if not part:
            continue

        is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/pcm")
        if is_audio:
            audio_data = part.inline_data and part.inline_data.data
            if audio_data:
                message = {
                    "mime_type": "audio/pcm",
                    "data": base64.b64encode(audio_data).decode("ascii")
                }
                yield f"data: {json.dumps(message)}\n\n"
                continue

        if part.text and event.partial:
            message = {
                "mime_type": "text/plain",
                "data": part.text
            }
            yield f"data: {json.dumps(message)}\n\n"

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
        if user_id_str in active_sessions:
            del active_sessions[user_id_str]

    async def event_generator():
        try:
            async for data in agent_to_client_sse(live_events):
                yield data
        except Exception as e:
            print(f"[SSE Error]: {e}")
        finally:
            cleanup()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

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

# # main.py

# import os
# import json
# import base64
# import warnings

# from dotenv import load_dotenv
# from google.genai.types import Part, Content, Blob
# from google.adk.runners import InMemoryRunner
# from google.adk.agents import LiveRequestQueue
# from google.adk.agents.run_config import RunConfig

# from fastapi import FastAPI, Request
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware

# from agent import root_agent

# warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# load_dotenv()

# APP_NAME = "ADK Streaming example"
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Configure for security in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Store active sessions
# active_sessions = {}

# async def start_agent_session(user_id, is_audio=False):
#     runner = InMemoryRunner(app_name=APP_NAME, agent=root_agent)
#     session = await runner.session_service.create_session(app_name=APP_NAME, user_id=user_id)
#     modality = "AUDIO" if is_audio else "TEXT"
#     run_config = RunConfig(response_modalities=[modality])
#     live_request_queue = LiveRequestQueue()
#     live_events = runner.run_live(session=session, live_request_queue=live_request_queue, run_config=run_config)
#     return live_events, live_request_queue

# async def agent_to_client_sse(live_events):
#     async for event in live_events:
#         if event.turn_complete or event.interrupted:
#             message = {
#                 "turn_complete": event.turn_complete,
#                 "interrupted": event.interrupted,
#             }
#             yield f"data: {json.dumps(message)}\n\n"
#             continue

#         part = event.content and event.content.parts and event.content.parts[0]
#         if not part:
#             continue

#         is_audio = part.inline_data and part.inline_data.mime_type.startswith("audio/pcm")
#         if is_audio:
#             audio_data = part.inline_data and part.inline_data.data
#             if audio_data:
#                 message = {
#                     "mime_type": "audio/pcm",
#                     "data": base64.b64encode(audio_data).decode("ascii")
#                 }
#                 yield f"data: {json.dumps(message)}\n\n"
#                 continue

#         if part.text and event.partial:
#             message = {
#                 "mime_type": "text/plain",
#                 "data": part.text
#             }
#             yield f"data: {json.dumps(message)}\n\n"

# @app.get("/")
# def root():
#     return {"message": "FastAPI ADK Server Running"}

# @app.get("/events/{user_id}")
# async def sse_endpoint(user_id: int, is_audio: str = "false"):
#     user_id_str = str(user_id)
#     live_events, live_request_queue = await start_agent_session(user_id_str, is_audio == "true")
#     active_sessions[user_id_str] = live_request_queue
#     print(f"Client #{user_id} connected via SSE, audio mode: {is_audio}")

#     def cleanup():
#         live_request_queue.close()
#         if user_id_str in active_sessions:
#             del active_sessions[user_id_str]
#         print(f"Client #{user_id} disconnected from SSE")

#     async def event_generator():
#         try:
#             async for data in agent_to_client_sse(live_events):
#                 yield data
#         except Exception as e:
#             print(f"Error in SSE stream: {e}")
#         finally:
#             cleanup()

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "Access-Control-Allow-Origin": "*",
#             "Access-Control-Allow-Headers": "Cache-Control"
#         }
#     )

# @app.post("/send/{user_id}")
# async def send_message_endpoint(user_id: int, request: Request):
#     user_id_str = str(user_id)
#     live_request_queue = active_sessions.get(user_id_str)
#     if not live_request_queue:
#         return {"error": "Session not found"}

#     message = await request.json()
#     mime_type = message["mime_type"]
#     data = message["data"]

#     if mime_type == "text/plain":
#         content = Content(role="user", parts=[Part.from_text(text=data)])
#         live_request_queue.send_content(content=content)
#         print(f"[CLIENT TO AGENT]: {data}")

#     elif mime_type == "audio/pcm":
#         decoded_data = base64.b64decode(data)
#         live_request_queue.send_realtime(Blob(data=decoded_data, mime_type=mime_type))
#         print(f"[CLIENT TO AGENT]: audio/pcm: {len(decoded_data)} bytes")

#     else:
#         return {"error": f"Mime type not supported: {mime_type}"}

#     return {"status": "sent"}
