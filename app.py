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
import uvicorn
from starlette.websockets import WebSocketState

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
CHUNK_ALIGNMENT = 320  # Exotel requires chunks to be multiples of 320 bytes
CHUNK_MIN_SIZE = 3200  # 100ms minimum as per Exotel requirements
CHUNK_MAX_SIZE = 96000  # Slightly less than 100k max to be safe
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.5  # seconds of silence to trigger processing
MAX_PROCESSING_TIME = 8.0  # Maximum time for processing to avoid timeouts

# FastAPI app
app = FastAPI()

# History for the LLM (system prompt)
history = [{
    "role": "system",
    "content": """
നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്. താങ്കളുടെ ജോലി വയനാട്ടിലെ വൈത്തിരി പാർക്കിന്റെ സന്ദർശകരുമായി സൗഹൃദപരമായ രീതിയിൽ സംസാരിക്കുകയും, അവരുടെ സംശയങ്ങൾക്കും ചോദ്യങ്ങൾക്കും വ്യക്തമായ മറുപടികൾ നൽകുകയും ചെയ്യുകയാണ്. താങ്കൾക്ക് എല്ലാവിധ മറുപടികളും ത്യാജ്യരീതിയിൽ, വിനീതതയോടെ നൽകേണ്ടതാണ്.

വൈത്തിരി പാർക്ക് വയനാട്ടിലെ വൈത്തിരിയിലാണ് സ്ഥിതി ചെയ്യുന്നത്. മനോഹരമായ മലനിരകൾ, പച്ചക്കാടുകൾ, മഞ്ഞുമൂടിയ കാലാവസ്ഥ തുടങ്ങിയ പ്രകൃതിദൃശ്യങ്ങൾക്കിടയിൽ സ്ഥിതി ചെയ്യുന്ന ഈ പാർക്ക് സാഹസിക വിനോദത്തിനും കുടുംബസമേതം സമയം ചെലവിടുന്നതിനും അനുയോജ്യമായ സ്ഥലമാണ്.

ഈ പാർക്കിൽ 40-ലധികം റൈഡുകൾ ലഭ്യമാണ്. അതിൽ അഡ്വഞ്ചർ റൈഡുകളും, അമ്യൂസ്മെന്റ് റൈഡുകളും, വാട്ടർ സ്ലൈഡുകളും ഉൾപ്പെടുന്നു. എല്ലാ പ്രായമുള്ള ആളുകൾക്കും അനുയോജ്യമായ രീതിയിലാണ് ആമസ്മെന്റ് ഒരുക്കങ്ങൾ രൂപകൽപ്പന ചെയ്തിരിക്കുന്നത്.

വൈത്തിരി പാർക്ക് ഓരോ ദിവസവും രാവിലെ 9 മണി മുതൽ വൈകിട്ട് 6 മണി വരെ തുറന്നിരിക്കുന്നു.

ടിക്കറ്റ് നിരക്കുകൾ ചുവടെ കാണാം:
- മുതിർന്നവർക്ക് ₹799 രൂപയാണ്.
- കുട്ടികൾക്ക് (ഉയരം 90 സെ.മി. മുതൽ 120 സെ.മി. വരെ) ₹599 രൂപയാണ്.
- മുതിർന്ന പൗരന്മാർക്ക് ₹300 രൂപയാണ്.

ഒരു സന്ദർശകനായി താങ്കൾ എന്തെങ്കിലും സംശയങ്ങൾ ഉന്നയിച്ചാൽ അതിന് ചുരുങ്ങിയതും വ്യക്തവുമായ മറുപടി നൽകേണ്ടതുണ്ട്. കൂടുതൽ വിശദീകരണം ആവശ്യമെങ്കിൽ, "കൂടുതൽ വിവരങ്ങൾക്ക് ഞങ്ങളുടെ സെൽസ് എക്സിക്യൂട്ടീവുമായി ബന്ധപ്പെടാമോ?" എന്ന് അകൃത്യമായും വിനീതമായും ചോദിക്കുക.

താങ്കളുടെ മറുപടികൾ എല്ലാം മലയാളത്തിലായിരിക്കണം. സന്ദർശകർക്ക് സഹായകരമായ ഉത്തരം നൽകാൻ ഉദ്ദേശിച്ചുകൊണ്ട് ചുരുങ്ങിയ വാക്കുകളിൽ സൗമ്യമായി പ്രതികരിക്കുക.
"""
}]

# -- Utility functions --
# Utility functions for audio processing and AI interactions
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

async def transcribe_pcm(pcm_data: bytes) -> str:
    temp_file = f"temp_{uuid.uuid4().hex}.wav"
    pcm_to_wav(pcm_data, temp_file)
    logger.info("Starting transcription")
    
    try:
        with open(temp_file, 'rb') as wf:
            result = sarvam_client.speech_to_text.transcribe(
                file=wf,
                model="saarika:v2.5",
                language_code="ml-IN"
            )
        transcript = result.transcript.strip()
        logger.info(f"Transcript: {transcript}")
        return transcript
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""
    finally:
        # Clean up temp file
        try:
            os.remove(temp_file)
        except Exception as e:
            logger.warning(f"Failed to remove temp file: {e}")

async def llm_respond(transcript: str) -> str:
    logger.info(f"LLM received: {transcript}")
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[:1] + history[-19:]  # Keep system prompt + last 19 messages
    
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.5,
            max_tokens=200  # Limit response length for faster processing
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ക്ഷമിക്കണം, തകരാറുണ്ടായി. വീണ്ടും ശ്രമിക്കാമോ?"

async def text_to_pcm(text: str) -> bytes:
    logger.info("Converting text to PCM via TTS")
    try:
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
    except Exception as e:
        logger.error(f"TTS error: {e}")
        # Generate a simple error message audio if TTS fails
        return b'\x00' * CHUNK_MIN_SIZE  # Silent audio as fallback

def chunk_pcm(pcm_data: bytes) -> list:
    """
    Chunk PCM data according to Exotel requirements:
    - Chunks must be multiples of 320 bytes
    - Minimum size: 3200 bytes (100ms)
    - Maximum size: ~100K bytes
    
    Returns a list of chunks instead of a generator for better control
    """
    if not pcm_data:
        return []
        
    chunks = []
    i = 0
    while i < len(pcm_data):
        # Calculate chunk size (min 3200, max ~96000, multiple of 320)
        end = min(i + CHUNK_MAX_SIZE, len(pcm_data))
        chunk = pcm_data[i:end]
        
        # Ensure multiple of 320 bytes
        remainder = len(chunk) % CHUNK_ALIGNMENT
        if remainder:
            chunk = chunk[:-remainder]
            
        # Skip chunks that are too small
        if len(chunk) < CHUNK_MIN_SIZE:
            break
            
        chunks.append(chunk)
        i += len(chunk)
        
    return chunks

async def safe_send_json(websocket, data):
    """Safely send JSON data over websocket with error handling"""
    if websocket.application_state != WebSocketState.CONNECTED:
        logger.warning("WebSocket not connected, can't send data")
        return False
        
    try:
        await websocket.send_json(data)
        return True
    except Exception as e:
        logger.error(f"Error sending data over websocket: {e}")
        return False

# -- WebSocket endpoint --
@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = deque()
    buffer_size = 0
    silence_start = None
    seq_num = 1
    last_activity = time.time()
    is_processing = False

    logger.info("WebSocket connection accepted")

    # Send heartbeats periodically to keep connection alive
    heartbeat_task = None
    
    async def send_heartbeat():
        nonlocal seq_num
        while True:
            try:
                if websocket.application_state == WebSocketState.CONNECTED:
                    current_time = time.time()
                    # Only send heartbeat if no activity for 5 seconds
                    if current_time - last_activity > 5 and stream_sid:
                        success = await safe_send_json(websocket, {
                            "event": "mark",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "mark": {"name": "heartbeat"}
                        })
                        if success:
                            logger.debug(f"Sent heartbeat, seq={seq_num}")
                            seq_num += 1
                    await asyncio.sleep(5)  # Check every 5 seconds
                else:
                    break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break
    
    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(send_heartbeat())
        
        while True:
            # Set a reasonable timeout for receive_json
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
                last_activity = time.time()
            except asyncio.TimeoutError:
                logger.warning("No message received for 10 seconds, sending heartbeat")
                if stream_sid:
                    await safe_send_json(websocket, {
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "heartbeat"}
                    })
                    seq_num += 1
                continue
                
            event = msg.get("event")
            logger.debug(f"Received event: {event}")

            if event == "connected":
                logger.info("Connected event received")
                continue

            if event == "start":
                stream_sid = msg.get("stream_sid")
                logger.info(f"Stream started: {stream_sid}")
                
                # Send welcome message
                welcome_text = "ഹലോ! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. എങ്ങനെ സഹായിക്കാൻ കഴിയും?"
                welcome_pcm = await text_to_pcm(welcome_text)
                pcm_chunks = chunk_pcm(welcome_pcm)
                
                timestamp = str(int(time.time() * 1000))
                for idx, chunk in enumerate(pcm_chunks, 1):
                    success = await safe_send_json(websocket, {
                        "event": "media",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "media": {
                            "chunk": idx,
                            "timestamp": timestamp,
                            "payload": base64.b64encode(chunk).decode()
                        }
                    })
                    if not success:
                        break
                    seq_num += 1
                    await asyncio.sleep(0.05)  # Small delay between chunks
                
                # Send end-of-welcome mark
                await safe_send_json(websocket, {
                    "event": "mark",
                    "sequence_number": seq_num,
                    "stream_sid": stream_sid,
                    "mark": {"name": "welcome-complete"}
                })
                seq_num += 1
                continue

            if event == "media":
                if is_processing:
                    # Skip buffering if we're processing audio
                    continue
                    
                payload = base64.b64decode(msg["media"]["payload"])
                audio_buffer.append(payload)
                buffer_size += len(payload)
                logger.debug(f"Buffered media chunk, size={len(payload)}, total buffer={buffer_size}")

                # Check if we have enough audio data (at least 1 second)
                if buffer_size > SAMPLE_RATE * SAMPLE_WIDTH and not is_processing:
                    # Process if either silence detected or buffer is getting large
                    if is_silence(payload):
                        if silence_start is None:
                            silence_start = time.time()
                            logger.debug("Silence detected, starting timer")
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            # Process due to silence
                            is_processing = True
                    elif buffer_size > SAMPLE_RATE * SAMPLE_WIDTH * 5:  # > 5 seconds of audio
                        # Process due to large buffer
                        is_processing = True
                        
                    if is_processing:
                        # Process audio in a new task to not block WebSocket
                        asyncio.create_task(process_audio(websocket, stream_sid, audio_buffer, seq_num))
                        # Reset buffers
                        audio_buffer = deque()
                        buffer_size = 0
                        silence_start = None
                        is_processing = False

            if event == "stop":
                logger.info("Stop event received; closing connection")
                break
                
            # Handle mark event from Exotel
            if event == "mark":
                mark_name = msg.get("mark", {}).get("name", "")
                logger.info(f"Received mark event: {mark_name}")
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cancel heartbeat task
        if heartbeat_task:
            heartbeat_task.cancel()
            
        try:
            await websocket.close()
            logger.info("WebSocket closed gracefully")
        except RuntimeError as e:
            logger.warning(f"WebSocket already closed: {e}")

async def process_audio(websocket, stream_sid, audio_buffer, start_seq_num):
    """Process audio and send response in a separate task"""
    seq_num = start_seq_num
    
    try:
        pcm_data = b"".join(audio_buffer)
        if len(pcm_data) < CHUNK_MIN_SIZE:
            logger.warning(f"Audio buffer too small to process: {len(pcm_data)} bytes")
            return seq_num
            
        # Transcribe audio
        transcript = await transcribe_pcm(pcm_data)
        if not transcript:
            logger.warning("No transcript detected")
            return seq_num
            
        # Send thinking mark
        await safe_send_json(websocket, {
            "event": "mark",
            "sequence_number": seq_num,
            "stream_sid": stream_sid,
            "mark": {"name": "thinking"}
        })
        seq_num += 1
        
        # Get LLM response
        reply = await llm_respond(transcript)
        
        # Convert to audio
        pcm_reply = await text_to_pcm(reply)
        pcm_chunks = chunk_pcm(pcm_reply)
        
        if not pcm_chunks:
            logger.warning("No audio chunks generated")
            return seq_num
            
        # Send audio chunks
        timestamp = str(int(time.time() * 1000))
        for idx, chunk in enumerate(pcm_chunks, 1):
            success = await safe_send_json(websocket, {
                "event": "media",
                "sequence_number": seq_num,
                "stream_sid": stream_sid,
                "media": {
                    "chunk": idx,
                    "timestamp": timestamp,
                    "payload": base64.b64encode(chunk).decode()
                }
            })
            if not success:
                break
                
            seq_num += 1
            await asyncio.sleep(0.05)  # Small delay between chunks
        
        # Send end-of-response mark
        await safe_send_json(websocket, {
            "event": "mark",
            "sequence_number": seq_num,
            "stream_sid": stream_sid,
            "mark": {"name": "response-complete"}
        })
        seq_num += 1
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        
        # Try to send error message
        try:
            error_text = "ക്ഷമിക്കണം, തകരാറുണ്ടായി. വീണ്ടും ശ്രമിക്കാമോ?"
            error_pcm = await text_to_pcm(error_text)
            pcm_chunks = chunk_pcm(error_pcm)
            
            timestamp = str(int(time.time() * 1000))
            for idx, chunk in enumerate(pcm_chunks, 1):
                await safe_send_json(websocket, {
                    "event": "media",
                    "sequence_number": seq_num,
                    "stream_sid": stream_sid,
                    "media": {
                        "chunk": idx,
                        "timestamp": timestamp,
                        "payload": base64.b64encode(chunk).decode()
                    }
                })
                seq_num += 1
                await asyncio.sleep(0.05)
        except Exception:
            logger.error("Failed to send error message", exc_info=True)
    
    return seq_num

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)