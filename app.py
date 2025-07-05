from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from collections import deque
from sarvamai import SarvamAI
from groq import Groq
import uvicorn
from starlette.websockets import WebSocketState
import os
import uuid
import base64
import time
import wave
import logging
import asyncio
import numpy as np

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
SILENCE_DURATION = 2.0  # seconds of silence to trigger processing

# FastAPI app
app = FastAPI()

# Health endpoint
@app.get("/health")
async def health_check():
    try:
        assert sarvam_client is not None
        assert groq_client is not None
        return JSONResponse(status_code=200, content={"status": "ok"})
    except Exception:
        logger.error("Health check failed: AI clients not initialized")
        raise HTTPException(status_code=503, detail="Service Unavailable")

# History for the LLM (system prompt)
history = [{
    "role": "system",
    "content": """... trimmed for brevity ..."""
}]

# Utility functions
def is_silence(audio_bytes: bytes) -> bool:
    if not audio_bytes:
        return True
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio_np) == 0:
        return True
    rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))
    logger.debug(f"RMS: {rms}")
    return rms < SILENCE_THRESHOLD

def pcm_to_wav(pcm_data: bytes, wav_path: str) -> None:
    logger.debug(f"Writing PCM to WAV: {wav_path}")
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

def chunk_pcm(pcm_data: bytes, min_size=3200, max_size=100000, alignment=320):
    i = 0
    while i < len(pcm_data):
        end = min(i + max_size, len(pcm_data))
        chunk = pcm_data[i:end]
        trim = len(chunk) % alignment
        if trim:
            chunk = chunk[:-trim]
        if len(chunk) < min_size:
            break
        yield chunk
        i += len(chunk)

@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = deque()
    silence_start = None
    last_silence_chunk_sent = time.time()
    last_keepalive = time.time()
    seq_num = 1
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
                continue

            if event == "media":
                payload = base64.b64decode(msg["media"]["payload"])
                audio_buffer.append(payload)
                logger.debug(f"Buffered media chunk, size={len(payload)}")

                if is_silence(payload):
                    if silence_start is None:
                        silence_start = time.time()
                        logger.debug("Silence detected, starting timer")
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        pcm_data = b"".join(audio_buffer)
                        audio_buffer.clear()
                        silence_start = None
                        logger.info("Processing buffered audio after silence")

                        transcript = transcribe_pcm(pcm_data)
                        if not transcript:
                            continue

                        reply = llm_respond(transcript)

                        if websocket.application_state == WebSocketState.CONNECTED:
                            await websocket.send_json({
                                "event": "mark",
                                "sequence_number": seq_num,
                                "stream_sid": stream_sid,
                                "mark": {"name": "thinking"}
                            })
                            logger.debug(f"Sent keep-alive mark event, seq={seq_num}")
                            seq_num += 1

                        try:
                            pcm_reply = await asyncio.to_thread(text_to_pcm, reply)
                        except Exception as e:
                            logger.error(f"TTS failed: {e}")
                            pcm_reply = await asyncio.to_thread(text_to_pcm, "ക്ഷമിക്കണം, തകരാറുണ്ടായി.")

                        timestamp = str(int(time.time() * 1000))
                        chunk_idx = 1
                        chunk_sent = False
                        for chunk in chunk_pcm(pcm_reply):
                            payload_b64 = base64.b64encode(chunk).decode()
                            await websocket.send_json({
                                "event": "media",
                                "sequence_number": seq_num,
                                "stream_sid": stream_sid,
                                "media": {
                                    "chunk": chunk_idx,
                                    "timestamp": timestamp,
                                    "payload": payload_b64
                                }
                            })
                            logger.debug(f"Sent media chunk {chunk_idx}, seq={seq_num}, size={len(chunk)}")
                            seq_num += 1
                            chunk_idx += 1
                            chunk_sent = True
                            await asyncio.sleep(0.1)

                        if chunk_sent:
                            await websocket.send_json({
                                "event": "mark",
                                "sequence_number": seq_num,
                                "stream_sid": stream_sid,
                                "mark": {"name": "end-of-reply"}
                            })
                            logger.info(f"Sent mark event, seq={seq_num}")
                            seq_num += 1
                else:
                    silence_start = None

                # Send silence chunk if prolonged silence
                if time.time() - silence_start > 2.0 and (time.time() - last_silence_chunk_sent > 2.0):
                    silence_chunk = b"\x00" * 3200
                    payload_b64 = base64.b64encode(silence_chunk).decode()
                    await websocket.send_json({
                        "event": "media",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "media": {
                            "chunk": 999,
                            "timestamp": str(int(time.time() * 1000)),
                            "payload": payload_b64
                        }
                    })
                    logger.debug(f"Sent keep-alive silence chunk, seq={seq_num}")
                    seq_num += 1
                    last_silence_chunk_sent = time.time()

                # Heartbeat mark event every 5 seconds
                if time.time() - last_keepalive > 5.0:
                    await websocket.send_json({
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "keep-alive"}
                    })
                    logger.debug(f"Sent keep-alive mark, seq={seq_num}")
                    seq_num += 1
                    last_keepalive = time.time()

            if event == "stop":
                logger.info("Stop event received; closing connection")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket ended.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
