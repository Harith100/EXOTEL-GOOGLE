import os
import time
import base64
import uuid
import wave
import asyncio
import numpy as np
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from collections import deque
from sarvamai import SarvamAI
from groq import Groq

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s:%(message)s',
)
logger = logging.getLogger("exotel_bot")

# Load environment variables
load_dotenv()
SARVAMAI_API_KEY = os.getenv("SARVAMAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize AI clients
sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Audio configuration
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_ALIGNMENT = 320
CHUNK_MIN_SIZE = 3200  # 100ms
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0  # seconds
MAX_AUDIO_BUFFER_DURATION = 5.0  # seconds

# FastAPI app
app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        assert sarvam_client is not None
        assert groq_client is not None
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception:
        logger.error("Health check failed: AI clients not initialized")
        raise HTTPException(status_code=503, detail="Service Unavailable")

history = [{
    "role": "system",
    "content": """
നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്. താങ്കളുടെ ജോലി വയനാട്ടിലെ വൈത്തിരി പാർക്കിന്റെ സന്ദർശകരുമായി സൗഹൃദപരമായ രീതിയിൽ സംസാരിക്കുകയും, അവരുടെ സംശയങ്ങൾക്കും ചോദ്യങ്ങൾക്കും വ്യക്തമായ മറുപടികൾ നൽകുകയും ചെയ്യുകയാണ്...
"""
}]

def is_silence(audio_bytes: bytes) -> bool:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_np ** 2))
    logger.debug(f"RMS: {rms}")
    return rms < SILENCE_THRESHOLD

def pcm_to_wav(pcm_data: bytes, wav_path: str) -> None:
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

def transcribe_pcm(pcm_data: bytes) -> str:
    temp_file = f"temp_{uuid.uuid4().hex}.wav"
    pcm_to_wav(pcm_data, temp_file)
    logger.info("Starting transcription")
    with open(temp_file, 'rb') as wf:
        result = sarvam_client.speech_to_text.transcribe(
            file=wf,
            model="saarika:v2.5",
            language_code="ml-IN"
        )
    os.remove(temp_file)
    transcript = result.transcript.strip()
    logger.info(f"Transcript: {transcript}")
    return transcript

def llm_respond(transcript: str) -> str:
    logger.info(f"LLM received: {transcript}")
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[-20:]
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=history,
        temperature=0.5
    )
    reply = resp.choices[0].message.content.strip()
    logger.info(f"LLM reply: {reply}")
    history.append({"role": "assistant", "content": reply})
    return reply

def text_to_pcm(text: str) -> bytes:
    logger.info("Converting text to PCM via TTS")
    resp = sarvam_client.text_to_speech.convert(
        text=text,
        target_language_code="ml-IN",
        speaker="manisha",
        enable_preprocessing=True,
        speech_sample_rate=SAMPLE_RATE
    )
    pcm = b"".join(base64.b64decode(chunk) for chunk in resp.audios)
    logger.debug(f"Generated PCM length: {len(pcm)}")
    return pcm

def chunk_pcm(pcm_data: bytes, size: int = CHUNK_MIN_SIZE):
    for i in range(0, len(pcm_data), size):
        chunk = pcm_data[i:i+size]
        trim = len(chunk) % CHUNK_ALIGNMENT
        if trim:
            chunk = chunk[:-trim]
        if chunk:
            yield chunk

@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = deque()
    silence_start = None
    audio_start_time = None
    seq_num = 1
    sent_initial_reply = False

    logger.info("WebSocket connection accepted")

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")
            logger.debug(f"Received event: {event}")

            if event == "connected":
                logger.info("Connected event received")
                continue

            if event == "start":
                stream_sid = msg.get("stream_sid")
                logger.info(f"Stream started: {stream_sid}")

                # Send welcome message
                welcome_text = "വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം! സഹായിക്കാൻ ഞാൻ ഇവിടെ തയ്യാറാണ്."
                welcome_pcm = text_to_pcm(welcome_text)
                timestamp = str(int(time.time() * 1000))
                chunk_idx = 1
                for chunk in chunk_pcm(welcome_pcm):
                    await websocket.send_json({
                        "event": "media",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "media": {
                            "chunk": chunk_idx,
                            "timestamp": timestamp,
                            "payload": base64.b64encode(chunk).decode()
                        }
                    })
                    logger.debug(f"Sent welcome chunk {chunk_idx}, seq={seq_num}")
                    seq_num += 1
                    chunk_idx += 1
                    await asyncio.sleep(0.1)

                await websocket.send_json({
                    "event": "mark",
                    "sequence_number": seq_num,
                    "stream_sid": stream_sid,
                    "mark": {"name": "welcome-sent"}
                })
                seq_num += 1
                continue

            if event == "media":
                payload = base64.b64decode(msg["media"]["payload"])
                audio_buffer.append(payload)
                logger.debug(f"Buffered media chunk, size={len(payload)}")

                if not sent_initial_reply:
                    logger.info("Sending initial quick TTS reply to keep session alive")
                    quick_reply_pcm = text_to_pcm("ഒരു സെക്കൻഡ്, ഞാൻ പരിശോധിക്കുന്നു.")
                    timestamp = str(int(time.time() * 1000))
                    chunk_idx = 1
                    for chunk in chunk_pcm(quick_reply_pcm):
                        await websocket.send_json({
                            "event": "media",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "media": {
                                "chunk": chunk_idx,
                                "timestamp": timestamp,
                                "payload": base64.b64encode(chunk).decode()
                            }
                        })
                        logger.debug(f"Sent quick keepalive chunk {chunk_idx}, seq={seq_num}")
                        seq_num += 1
                        chunk_idx += 1
                        await asyncio.sleep(0.1)
                    sent_initial_reply = True

                if audio_start_time is None:
                    audio_start_time = time.time()

                buffer_duration = time.time() - audio_start_time
                silence = is_silence(payload)
                process_audio = False

                if silence:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        process_audio = True
                elif buffer_duration >= MAX_AUDIO_BUFFER_DURATION:
                    process_audio = True

                if process_audio:
                    pcm_data = b"".join(audio_buffer)
                    audio_buffer.clear()
                    silence_start = None
                    audio_start_time = None

                    transcript = transcribe_pcm(pcm_data)
                    if not transcript:
                        continue

                    reply = llm_respond(transcript)
                    pcm_reply = text_to_pcm(reply)

                    timestamp = str(int(time.time() * 1000))
                    chunk_idx = 1
                    for chunk in chunk_pcm(pcm_reply):
                        await websocket.send_json({
                            "event": "media",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "media": {
                                "chunk": chunk_idx,
                                "timestamp": timestamp,
                                "payload": base64.b64encode(chunk).decode()
                            }
                        })
                        logger.debug(f"Sent media chunk {chunk_idx}, seq={seq_num}")
                        seq_num += 1
                        chunk_idx += 1
                        await asyncio.sleep(0.1)

                    await websocket.send_json({
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "end-of-reply"}
                    })
                    logger.info(f"Sent mark event, seq={seq_num}")
                    seq_num += 1

            if event == "stop":
                logger.info("Stop event received; closing connection")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        logger.info("WebSocket closed")
