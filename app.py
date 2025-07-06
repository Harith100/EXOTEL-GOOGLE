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
നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്... [TRUNCATED FOR BREVITY]
"""
}]

# ---------------- UTILS ----------------

def is_silence(audio_bytes: bytes) -> bool:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_np ** 2))
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

def chunk_pcm(pcm_data: bytes, size: int = CHUNK_MIN_SIZE):
    for i in range(0, len(pcm_data), size):
        chunk = pcm_data[i:i+size]
        trim = len(chunk) % CHUNK_ALIGNMENT
        if trim:
            chunk = chunk[:-trim]
        if chunk:
            yield chunk

# ---------------- WebSocket ----------------

@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = deque()
    silence_start = None
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

                welcome_text = "വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം! ഞാൻ നിങ്ങളെ സഹായിക്കാനാണ് ഇവിടെ."
                pcm_welcome = text_to_pcm(welcome_text)

                chunk_idx = 1
                for chunk in chunk_pcm(pcm_welcome):
                    try:
                        await websocket.send_json({
                            "event": "media",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "media": {
                                "chunk": chunk_idx,
                                "timestamp": str(int(time.time() * 1000)),
                                "payload": base64.b64encode(chunk).decode()
                            }
                        })
                        logger.debug(f"Sent welcome chunk {chunk_idx}, seq={seq_num}")
                        seq_num += 1
                        chunk_idx += 1
                        await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"WebSocket send failed (welcome): {e}")
                        break

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
                        pcm_reply = text_to_pcm(reply)

                        timestamp = str(int(time.time() * 1000))
                        chunk_idx = 1
                        for chunk in chunk_pcm(pcm_reply):
                            try:
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
                                logger.debug(f"Sent reply chunk {chunk_idx}, seq={seq_num}")
                                seq_num += 1
                                chunk_idx += 1
                                await asyncio.sleep(0.1)
                            except Exception as e:
                                logger.warning(f"WebSocket send failed (reply): {e}")
                                break

                        try:
                            await websocket.send_json({
                                "event": "mark",
                                "sequence_number": seq_num,
                                "stream_sid": stream_sid,
                                "mark": {"name": "end-of-reply"}
                            })
                            logger.info(f"Sent mark event, seq={seq_num}")
                            seq_num += 1
                        except Exception as e:
                            logger.warning(f"Failed to send mark event: {e}")

            if event == "stop":
                logger.info("Stop event received; closing connection")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        if websocket.client_state.name == "CONNECTED":
            await websocket.close()
            logger.info("WebSocket closed safely")
        else:
            logger.info("WebSocket already closed")
