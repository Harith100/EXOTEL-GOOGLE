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
CHUNK_MAX_SIZE = 32000  # Smaller max size for quicker processing
SILENCE_THRESHOLD = 300  # Lower threshold to detect silence more aggressively
SILENCE_DURATION = 0.5  # Shorter silence duration to trigger processing
PROCESS_INTERVAL = 1.0  # Process every second even without silence

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
    if not transcript:
        return "വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. എങ്ങനെ സഹായിക്കാൻ കഴിയും?"
        
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[:1] + history[-19:]  # Keep system prompt + last 19 messages
    
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.5,
            max_tokens=150  # Keep responses short for faster processing
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല. വീണ്ടും പറയാമോ?"

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
    Chunk PCM data according to Exotel requirements
    """
    if not pcm_data or len(pcm_data) < CHUNK_MIN_SIZE:
        return []
        
    chunks = []
    i = 0
    while i < len(pcm_data):
        # Calculate chunk size (min 3200, max CHUNK_MAX_SIZE, multiple of 320)
        chunk_size = min(CHUNK_MAX_SIZE, len(pcm_data) - i)
        chunk_size = chunk_size - (chunk_size % CHUNK_ALIGNMENT)
        
        if chunk_size < CHUNK_MIN_SIZE:
            break
            
        chunks.append(pcm_data[i:i+chunk_size])
        i += chunk_size
        
    return chunks

async def safe_send_json(websocket, data):
    """Safely send JSON data over websocket with error handling"""
    try:
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
        return False
    except Exception as e:
        logger.error(f"Error sending data over websocket: {e}")
        return False

async def send_audio_response(websocket, stream_sid, seq_num, text):
    """Send an audio response to the user"""
    if not text or not stream_sid:
        return seq_num
        
    try:
        # Convert text to audio
        pcm_data = await text_to_pcm(text)
        chunks = chunk_pcm(pcm_data)
        
        if not chunks:
            logger.warning("No audio chunks to send")
            return seq_num
            
        # Send audio chunks
        timestamp = str(int(time.time() * 1000))
        for idx, chunk in enumerate(chunks, 1):
            if websocket.application_state != WebSocketState.CONNECTED:
                logger.warning("WebSocket disconnected while sending audio")
                break
                
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
            # Very small delay to avoid overwhelming Exotel
            await asyncio.sleep(0.01)
            
        # Send end-of-response mark
        if websocket.application_state == WebSocketState.CONNECTED:
            await safe_send_json(websocket, {
                "event": "mark",
                "sequence_number": seq_num,
                "stream_sid": stream_sid,
                "mark": {"name": "response-complete"}
            })
            seq_num += 1
            
    except Exception as e:
        logger.error(f"Error sending audio response: {e}")
        
    return seq_num

# -- WebSocket endpoint --
@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = bytearray()
    last_process_time = time.time()
    silence_start = None
    seq_num = 1
    processing_task = None
    is_closing = False

    logger.info("WebSocket connection accepted")

    async def process_audio_buffer():
        nonlocal seq_num, audio_buffer
        
        if len(audio_buffer) < CHUNK_MIN_SIZE:
            logger.debug("Audio buffer too small to process")
            return
            
        # Make a copy of the buffer and clear the original
        pcm_data = bytes(audio_buffer)
        audio_buffer.clear()
        
        # Quick acknowledgment to keep connection alive
        if stream_sid and websocket.application_state == WebSocketState.CONNECTED:
            await safe_send_json(websocket, {
                "event": "mark",
                "sequence_number": seq_num,
                "stream_sid": stream_sid,
                "mark": {"name": "processing"}
            })
            seq_num += 1
        
        # Process the audio
        transcript = await transcribe_pcm(pcm_data)
        if not transcript:
            logger.warning("No transcript detected")
            return
            
        # Get LLM response
        reply = await llm_respond(transcript)
        
        # Send audio response
        if stream_sid and websocket.application_state == WebSocketState.CONNECTED:
            seq_num = await send_audio_response(websocket, stream_sid, seq_num, reply)

    try:
        # Send welcome message once connected
        welcome_sent = False
        
        while not is_closing:
            try:
                # Short timeout to regularly check buffer
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.5)
                event = msg.get("event")
                logger.debug(f"Received event: {event}")

                if event == "connected":
                    logger.info("Connected event received")
                    continue

                if event == "start":
                    stream_sid = msg.get("stream_sid")
                    logger.info(f"Stream started: {stream_sid}")
                    
                    # Send welcome message
                    if not welcome_sent and stream_sid:
                        welcome_text = "ഹലോ! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. എങ്ങനെ സഹായിക്കാൻ കഴിയും?"
                        seq_num = await send_audio_response(websocket, stream_sid, seq_num, welcome_text)
                        welcome_sent = True
                    continue

                if event == "media":
                    payload = base64.b64decode(msg["media"]["payload"])
                    audio_buffer.extend(payload)
                    logger.debug(f"Buffered media chunk, size={len(payload)}, total buffer={len(audio_buffer)}")
                    
                    current_time = time.time()
                    should_process = False
                    
                    # Check for silence to trigger processing
                    if is_silence(payload):
                        if silence_start is None:
                            silence_start = current_time
                        elif current_time - silence_start >= SILENCE_DURATION:
                            should_process = True
                    else:
                        silence_start = None
                    
                    # Process based on time interval or buffer size
                    if (current_time - last_process_time >= PROCESS_INTERVAL and len(audio_buffer) >= CHUNK_MIN_SIZE) or \
                       len(audio_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH * 3:  # Process after 3 seconds of audio
                        should_process = True
                    
                    # Start processing if needed
                    if should_process and len(audio_buffer) >= CHUNK_MIN_SIZE:
                        last_process_time = current_time
                        silence_start = None
                        
                        # Cancel any existing processing task
                        if processing_task and not processing_task.done():
                            processing_task.cancel()
                            
                        # Start new processing task
                        processing_task = asyncio.create_task(process_audio_buffer())

                if event == "stop":
                    logger.info("Stop event received")
                    is_closing = True
                    
                    # Process any remaining audio before closing
                    if len(audio_buffer) >= CHUNK_MIN_SIZE:
                        await process_audio_buffer()
                    break

                # Handle mark events from Exotel
                if event == "mark":
                    mark_name = msg.get("mark", {}).get("name", "")
                    logger.info(f"Received mark event: {mark_name}")

            except asyncio.TimeoutError:
                # Check buffer regularly during timeout
                current_time = time.time()
                if len(audio_buffer) >= CHUNK_MIN_SIZE and current_time - last_process_time >= PROCESS_INTERVAL:
                    last_process_time = current_time
                    await process_audio_buffer()
                continue

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Cancel any ongoing processing
        if processing_task and not processing_task.done():
            processing_task.cancel()
            
        # Clean close
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
            logger.info("WebSocket closed gracefully")
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")

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