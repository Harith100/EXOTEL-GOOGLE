import os
import time
import wave
import uuid
import base64
import asyncio
import logging
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from starlette.websockets import WebSocketState
from collections import deque
from sarvamai import SarvamAI
from groq import Groq

# Logging setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s:%(message)s')
logger = logging.getLogger("exotel_bot")

# Load .env
load_dotenv()
SARVAMAI_API_KEY = os.getenv("SARVAMAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Init clients
sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Audio config
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_ALIGNMENT = 320
CHUNK_MIN_SIZE = 3200
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0

app = FastAPI()

# Initial system message for LLM
history = [{
    "role": "system",
    "content": """നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്..."""  # Your full prompt
}]

@app.get("/health")
async def health_check():
    try:
        assert sarvam_client and groq_client
        return JSONResponse(status_code=200, content={"status": "ok"})
    except:
        raise HTTPException(status_code=503, detail="Service Unavailable")

# Helper functions
def is_silence(audio_bytes: bytes) -> bool:
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    if len(audio_np) == 0:
        return True
    rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))
    return rms < SILENCE_THRESHOLD

def pcm_to_wav(pcm_data: bytes, wav_path: str):
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)

def transcribe_pcm(pcm_data: bytes) -> str:
    temp_file = f"temp_{uuid.uuid4().hex}.wav"
    pcm_to_wav(pcm_data, temp_file)
    with open(temp_file, 'rb') as wf:
        result = sarvam_client.speech_to_text.transcribe(
            file=wf,
            model="saarika:v2.5",
            language_code="ml-IN"
        )
    os.remove(temp_file)
    return result.transcript.strip()

def llm_respond(transcript: str) -> str:
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[-20:]
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=history,
        temperature=0.5
    )
    reply = resp.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply})
    return reply

def text_to_pcm(text: str) -> bytes:
    resp = sarvam_client.text_to_speech.convert(
        text=text,
        target_language_code="ml-IN",
        speaker="manisha",
        enable_preprocessing=True,
        speech_sample_rate=SAMPLE_RATE
    )
    return b"".join(base64.b64decode(chunk) for chunk in resp.audios)

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
    seq_num = 1
    last_keepalive_sent = 0

    logger.info("WebSocket connection accepted")

    try:
        while True:
            msg = await websocket.receive_json()
            event = msg.get("event")

            if event == "connected":
                logger.info("connected")
                continue
            if event == "start":
                stream_sid = msg.get("stream_sid")
                logger.info(f"Stream started: {stream_sid}")
                continue
            if event == "media":
                payload = base64.b64decode(msg["media"]["payload"])
                audio_buffer.append(payload)

                # Send keep-alive if silence persists
                if is_silence(payload):
                    now = time.time()
                    if silence_start is None:
                        silence_start = now
                    elif now - silence_start >= SILENCE_DURATION:
                        pcm_data = b"".join(audio_buffer)
                        audio_buffer.clear()
                        silence_start = None

                        transcript = transcribe_pcm(pcm_data)
                        if not transcript:
                            continue

                        reply = llm_respond(transcript)

                        # Optional: "thinking" mark
                        await websocket.send_json({
                            "event": "mark",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "mark": {"name": "thinking"}
                        })
                        seq_num += 1

                        try:
                            pcm_reply = await asyncio.to_thread(text_to_pcm, reply)
                        except Exception:
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
                            seq_num += 1
                else:
                    # Reset silence timer
                    silence_start = None

                    # Keep connection alive during talking silence
                    if time.time() - last_keepalive_sent > 2:
                        await websocket.send_json({
                            "event": "mark",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "mark": {"name": "keep-alive"}
                        })
                        seq_num += 1
                        last_keepalive_sent = time.time()

            if event == "stop":
                logger.info("Stop received. Closing.")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
                logger.info("WebSocket closed gracefully")
        except Exception as e:
            logger.warning(f"WebSocket already closed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
