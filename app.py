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
from starlette.websockets import WebSocketState, WebSocketDisconnect

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
CHUNK_MAX_SIZE = 6400  # Smaller chunks for faster streaming
SILENCE_THRESHOLD = 300  # Lower threshold to detect silence more aggressively
SILENCE_DURATION = 0.5  # Seconds of silence to trigger processing
MAX_BUFFER_SIZE = 48000  # Max buffer size (6 seconds of audio at 8KHz)
TIMEOUT_LIMIT = 3.0  # Maximum time to wait for response before sending interim messages

# FastAPI app
app = FastAPI()

# Pre-generated quick responses for fast acknowledgment
QUICK_RESPONSES = {
    "thinking": "ഒരു നിമിഷം കാത്തിരിക്കൂ...",
    "processing": "ഞാൻ നിങ്ങളുടെ ചോദ്യം പരിശോധിക്കുകയാണ്...",
    "error": "ക്ഷമിക്കണം, എന്തോ തകരാറുണ്ടായി. വീണ്ടും ശ്രമിക്കാമോ?",
    "welcome": "ഹലോ! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. എങ്ങനെ സഹായിക്കാൻ കഴിയും?"
}

# Cache for pre-generated audio PCM data
pcm_cache = {}

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
        return QUICK_RESPONSES["welcome"]
        
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[:1] + history[-19:]  # Keep system prompt + last 19 messages
    
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.5,
            max_tokens=80  # Keep responses shorter for faster processing
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return QUICK_RESPONSES["error"]

async def get_cached_pcm(text_key):
    """Get PCM data from cache or generate it"""
    if text_key in pcm_cache:
        return pcm_cache[text_key]
    
    text = QUICK_RESPONSES.get(text_key, text_key)
    pcm_data = await text_to_pcm(text)
    pcm_cache[text_key] = pcm_data
    return pcm_data

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
        # Return silent audio as fallback
        return b'\x00' * CHUNK_MIN_SIZE

def align_chunk_size(size):
    """Align chunk size to be a multiple of 320 bytes"""
    return size - (size % CHUNK_ALIGNMENT)

def chunk_pcm(pcm_data: bytes) -> list:
    """
    Chunk PCM data according to Exotel requirements:
    - Chunks must be multiples of 320 bytes
    - Minimum size: 3200 bytes (100ms)
    - Maximum size: CHUNK_MAX_SIZE bytes
    """
    if not pcm_data or len(pcm_data) < CHUNK_MIN_SIZE:
        return []
        
    chunks = []
    i = 0
    while i < len(pcm_data):
        # Calculate chunk size (min 3200, max CHUNK_MAX_SIZE, multiple of 320)
        chunk_size = min(CHUNK_MAX_SIZE, len(pcm_data) - i)
        chunk_size = align_chunk_size(chunk_size)
        
        if chunk_size < CHUNK_MIN_SIZE:
            # If remaining data is less than minimum, append to previous chunk
            # if possible, otherwise discard
            if chunks and i + chunk_size <= len(pcm_data):
                last_chunk = chunks.pop()
                chunks.append(last_chunk + pcm_data[i:i+chunk_size])
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

async def send_thinking_message(websocket, stream_sid, seq_num):
    """Send a quick thinking message to keep the connection alive"""
    if websocket.application_state != WebSocketState.CONNECTED or not stream_sid:
        return seq_num
        
    try:
        # Pre-generate thinking message PCM once
        if "thinking" not in pcm_cache:
            pcm_cache["thinking"] = await text_to_pcm(QUICK_RESPONSES["thinking"])
            
        pcm_data = pcm_cache["thinking"]
        chunks = chunk_pcm(pcm_data)
        
        if chunks:
            timestamp = str(int(time.time() * 1000))
            
            # First send a mark indicating we're thinking
            await safe_send_json(websocket, {
                "event": "mark",
                "sequence_number": seq_num,
                "stream_sid": stream_sid,
                "mark": {"name": "thinking"}
            })
            seq_num += 1
            
            # Send first chunk immediately
            if chunks:
                await safe_send_json(websocket, {
                    "event": "media",
                    "sequence_number": seq_num,
                    "stream_sid": stream_sid,
                    "media": {
                        "chunk": 1,
                        "timestamp": timestamp,
                        "payload": base64.b64encode(chunks[0]).decode()
                    }
                })
                seq_num += 1
                
                # End with a mark to keep the connection alive
                await safe_send_json(websocket, {
                    "event": "mark",
                    "sequence_number": seq_num,
                    "stream_sid": stream_sid,
                    "mark": {"name": "thinking-sent"}
                })
                seq_num += 1
    except Exception as e:
        logger.error(f"Error sending thinking message: {e}")
        
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
    welcome_sent = False
    is_processing = False

    logger.info("WebSocket connection accepted")
    
    # Pre-generate welcome message on startup to reduce initial latency
    asyncio.create_task(get_cached_pcm("welcome"))

    try:
        while True:
            try:
                # Short timeout for receiving messages
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.2)
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
                        # Use cached welcome message
                        welcome_pcm = await get_cached_pcm("welcome")
                        
                        # Split PCM data into smaller chunks to ensure consistent delivery
                        chunks = chunk_pcm(welcome_pcm)
                        
                        if chunks:
                            # Send each chunk with minimal delay
                            timestamp = str(int(time.time() * 1000))
                            
                            # Send chunks in smaller batches with minimal delay
                            for idx, chunk in enumerate(chunks, 1):
                                if websocket.application_state != WebSocketState.CONNECTED:
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
                                # Minimal delay between chunks
                                await asyncio.sleep(0.01)
                            
                            # Send mark to indicate welcome is complete
                            if websocket.application_state == WebSocketState.CONNECTED:
                                await safe_send_json(websocket, {
                                    "event": "mark",
                                    "sequence_number": seq_num,
                                    "stream_sid": stream_sid,
                                    "mark": {"name": "welcome-complete"}
                                })
                                seq_num += 1
                                welcome_sent = True
                    continue

                if event == "media":
                    payload = base64.b64decode(msg["media"]["payload"])
                    audio_buffer.extend(payload)
                    logger.debug(f"Buffered media chunk, size={len(payload)}, total buffer={len(audio_buffer)}")
                    
                    # Process immediately if we've received enough audio and 
                    # we're not already processing
                    current_time = time.time()
                    should_process = False
                    
                    # Process if we detect silence after speech
                    if is_silence(payload):
                        if silence_start is None:
                            silence_start = current_time
                        elif current_time - silence_start >= SILENCE_DURATION:
                            should_process = True
                    else:
                        silence_start = None
                    
                    # Also process if buffer gets large or too much time passed
                    if len(audio_buffer) >= MAX_BUFFER_SIZE or \
                       (len(audio_buffer) >= CHUNK_MIN_SIZE * 2 and current_time - last_process_time >= 2.0):
                        should_process = True
                    
                    # Start processing if needed and not already processing
                    if should_process and len(audio_buffer) >= CHUNK_MIN_SIZE and not is_processing:
                        # Make a copy of the buffer and clear the original
                        audio_data = bytes(audio_buffer)
                        audio_buffer.clear()
                        last_process_time = current_time
                        silence_start = None
                        
                        # Set processing flag
                        is_processing = True
                        
                        # Send thinking message immediately to keep connection alive
                        seq_num = await send_thinking_message(websocket, stream_sid, seq_num)
                        
                        # Process audio and respond in the background
                        task = asyncio.create_task(
                            process_and_respond(websocket, stream_sid, audio_data, seq_num)
                        )
                        
                        # Set up proper callback to reset the processing flag
                        def reset_processing_flag(future):
                            nonlocal is_processing
                            is_processing = False
                        
                        task.add_done_callback(reset_processing_flag)

                elif event == "mark":
                    # Handle mark events from Exotel
                    mark_name = msg.get("mark", {}).get("name", "")
                    logger.info(f"Received mark event: {mark_name}")
                
                elif event == "stop":
                    logger.info("Stop event received - closing connection")
                    break
                
            except asyncio.TimeoutError:
                # Check buffer periodically for automatic processing
                if not is_processing and len(audio_buffer) >= CHUNK_MIN_SIZE * 2 and time.time() - last_process_time >= 2.0:
                    audio_data = bytes(audio_buffer)
                    audio_buffer.clear()
                    last_process_time = time.time()
                    
                    # Set processing flag
                    is_processing = True
                    
                    # Send thinking message immediately
                    seq_num = await send_thinking_message(websocket, stream_sid, seq_num)
                    
                    # Process audio and respond in the background
                    task = asyncio.create_task(
                        process_and_respond(websocket, stream_sid, audio_data, seq_num)
                    )
                    
                    # Set up proper callback to reset the processing flag
                    def reset_processing_flag(future):
                        nonlocal is_processing
                        is_processing = False
                    
                    task.add_done_callback(reset_processing_flag)
                continue
            
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Clean close
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close()
            logger.info("WebSocket closed gracefully")
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")

async def process_and_respond(websocket, stream_sid, audio_data, start_seq_num):
    """Process audio data and send response with timeout handling"""
    seq_num = start_seq_num
    
    try:
        # Set up timeout tasks
        process_task = asyncio.create_task(process_audio(audio_data))
        
        # Wait for processing or timeout
        try:
            # Wait for processing with timeout
            transcript, reply = await asyncio.wait_for(process_task, timeout=TIMEOUT_LIMIT)
        except asyncio.TimeoutError:
            # If processing is taking too long, send a temporary response
            # but continue processing in the background
            logger.warning("Processing timeout - sending interim response")
            
            if websocket.application_state == WebSocketState.CONNECTED:
                # Send a message to keep the connection alive
                thinking_pcm = await get_cached_pcm("processing")
                chunks = chunk_pcm(thinking_pcm)
                
                if chunks:
                    timestamp = str(int(time.time() * 1000))
                    for idx, chunk in enumerate(chunks, 1):
                        if websocket.application_state != WebSocketState.CONNECTED:
                            break
                            
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
                        await asyncio.sleep(0.01)
                    
                    # Send mark to indicate interim response
                    await safe_send_json(websocket, {
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "interim-response-complete"}
                    })
                    seq_num += 1
            
            # Continue processing in the background
            transcript, reply = await process_task
        
        # Now send the actual response
        if websocket.application_state == WebSocketState.CONNECTED:
            # Convert response to audio
            pcm_reply = await text_to_pcm(reply)
            
            # Send in small chunks with minimal delay
            chunks = chunk_pcm(pcm_reply)
            if chunks:
                timestamp = str(int(time.time() * 1000))
                
                for idx, chunk in enumerate(chunks, 1):
                    if websocket.application_state != WebSocketState.CONNECTED:
                        break
                    
                    try:
                        await websocket.send_json({
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
                        
                        # Very minimal delay to avoid overwhelming Exotel
                        await asyncio.sleep(0.01)
                        
                    except Exception as e:
                        logger.error(f"Error sending audio chunk: {e}")
                        break
                
                # Send completion mark
                if websocket.application_state == WebSocketState.CONNECTED:
                    await safe_send_json(websocket, {
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "response-complete"}
                    })
                    seq_num += 1
        
    except Exception as e:
        logger.error(f"Error in process_and_respond: {e}", exc_info=True)
        
        # Try to send error message
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                error_pcm = await get_cached_pcm("error")
                chunks = chunk_pcm(error_pcm)
                
                if chunks:
                    timestamp = str(int(time.time() * 1000))
                    for idx, chunk in enumerate(chunks, 1):
                        await websocket.send_json({
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
                        await asyncio.sleep(0.01)
                    
                    await websocket.send_json({
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "error-message-complete"}
                    })
                    seq_num += 1
            except Exception:
                logger.error("Failed to send error message")
    
    return seq_num

async def process_audio(audio_data):
    """Process audio data and return transcript and reply"""
    # Transcribe the audio
    transcript = await transcribe_pcm(audio_data)
    
    # Get LLM response
    reply = await llm_respond(transcript)
    
    return transcript, reply

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

# Preload cache on startup
@app.on_event("startup")
async def preload_cache():
    for key in QUICK_RESPONSES:
        asyncio.create_task(get_cached_pcm(key))
    logger.info("Preloading response cache")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)