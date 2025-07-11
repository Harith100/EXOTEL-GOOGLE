import os
import time
import base64
import uuid
import wave
import asyncio
import numpy as np
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import JSONResponse
from collections import deque
from sarvamai import SarvamAI
from groq import Groq
import io
import struct

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

# Audio configuration - CRITICAL: Match Exotel's exact format
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_SIZE = 1600  # 100ms at 8kHz = 800 samples * 2 bytes
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0  # seconds of silence to trigger processing
SEND_INTERVAL = 0.02  # 20ms intervals for smoother playback

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
    """Check if audio chunk is silence"""
    if len(audio_bytes) < 2:
        return True
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
    rms = np.sqrt(np.mean(audio_np ** 2))
    logger.debug(f"RMS: {rms}")
    return rms < SILENCE_THRESHOLD


def pcm_to_wav(pcm_data: bytes, wav_path: str) -> None:
    """Convert PCM to WAV file"""
    logger.debug(f"Writing PCM to WAV: {wav_path}, size: {len(pcm_data)}")
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)


def transcribe_pcm(pcm_data: bytes) -> str:
    """Transcribe PCM audio using Sarvam AI"""
    if len(pcm_data) < 1600:  # Skip if less than 100ms of audio
        return ""
    
    temp_file = f"temp_{uuid.uuid4().hex}.wav"
    try:
        pcm_to_wav(pcm_data, temp_file)
        logger.info("Starting transcription")
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
        if os.path.exists(temp_file):
            os.remove(temp_file)


def llm_respond(transcript: str) -> str:
    """Generate response using Groq LLM"""
    logger.info(f"LLM received: {transcript}")
    history.append({"role": "user", "content": transcript})
    
    # Keep history manageable
    if len(history) > 20:
        history[:] = [history[0]] + history[-19:]  # Keep system prompt + last 19 messages
    
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",  # Using a more stable model
            messages=history,
            temperature=0.5,
            max_tokens=150  # Keep responses concise
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "മാപ്പുണ്ട്, എനിക്ക് ഇപ്പോൾ നിങ്ങളെ സഹായിക്കാൻ കഴിയുന്നില്ല."


def text_to_pcm(text: str) -> bytes:
    """Convert text to PCM using Sarvam TTS"""
    logger.info("Converting text to PCM via TTS")
    try:
        resp = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code="ml-IN",
            speaker="manisha",
            enable_preprocessing=True,
            speech_sample_rate=SAMPLE_RATE
        )
        
        pcm_chunks = []
        for b64 in resp.audios:
            wav_bytes = base64.b64decode(b64)
            # Extract PCM from WAV
            with wave.open(io.BytesIO(wav_bytes), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                pcm_chunks.append(frames)
        
        pcm = b"".join(pcm_chunks)
        logger.debug(f"Generated PCM length: {len(pcm)}")
        return pcm
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return b""


def ensure_linear_pcm(audio_data: bytes) -> bytes:
    """Ensure audio is in correct Linear PCM format for Exotel"""
    # If data length is odd, pad with a zero byte
    if len(audio_data) % 2 != 0:
        audio_data += b'\x00'
    
    # Verify it's valid 16-bit PCM by checking sample values
    try:
        samples = np.frombuffer(audio_data, dtype=np.int16)
        # Ensure samples are in valid range
        samples = np.clip(samples, -32768, 32767)
        return samples.tobytes()
    except Exception as e:
        logger.error(f"PCM validation error: {e}")
        return audio_data


# -- WebSocket endpoint --
@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    
    # Session variables
    stream_sid = None
    audio_buffer = bytearray()
    silence_start = None
    seq_num = 1
    is_processing = False
    connection_active = True
    
    logger.info("WebSocket connection accepted")
    
    try:
        while connection_active:
            try:
                # Set timeout for receive to handle disconnections gracefully
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                event = msg.get("event")
                logger.debug(f"Received event: {event}")
                
                if event == "connected":
                    logger.info("Connected event received")
                    # Send initial greeting after connection
                    greeting = "നമസ്കാരം! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. ഞാൻ എങ്ങനെ സഹായിക്കാം?"
                    initial_pcm = text_to_pcm(greeting)
                    if initial_pcm and stream_sid:
                        await send_audio_to_exotel(websocket, initial_pcm, stream_sid, seq_num)
                        seq_num += len(initial_pcm) // CHUNK_SIZE + 1
                    continue
                
                elif event == "start":
                    stream_sid = msg.get("streamSid") or msg.get("stream_sid")
                    logger.info(f"Stream started: {stream_sid}")
                    
                    # Send initial greeting
                    greeting = "നമസ്കാരം! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. ഞാൻ എങ്ങനെ സഹായിക്കാം?"
                    initial_pcm = text_to_pcm(greeting)
                    if initial_pcm:
                        await send_audio_to_exotel(websocket, initial_pcm, stream_sid, seq_num)
                        seq_num += len(initial_pcm) // CHUNK_SIZE + 1
                    continue
                
                elif event == "media":
                    if is_processing:
                        continue  # Skip incoming audio while processing
                    
                    # Decode audio payload
                    payload = base64.b64decode(msg["media"]["payload"])
                    audio_buffer.extend(payload)
                    
                    # Check for silence
                    if is_silence(payload):
                        if silence_start is None:
                            silence_start = time.time()
                            logger.debug("Silence detected, starting timer")
                        elif time.time() - silence_start >= SILENCE_DURATION and len(audio_buffer) > 3200:
                            # Process accumulated audio
                            is_processing = True
                            pcm_data = bytes(audio_buffer)
                            audio_buffer.clear()
                            silence_start = None
                            
                            logger.info(f"Processing buffered audio, size: {len(pcm_data)}")
                            
                            # Transcribe
                            transcript = transcribe_pcm(pcm_data)
                            if transcript:
                                # Generate response
                                reply = llm_respond(transcript)
                                
                                # Convert to speech
                                pcm_reply = text_to_pcm(reply)
                                if pcm_reply:
                                    # Send audio response
                                    await send_audio_to_exotel(websocket, pcm_reply, stream_sid, seq_num)
                                    seq_num += len(pcm_reply) // CHUNK_SIZE + 1
                            
                            is_processing = False
                    else:
                        silence_start = None  # Reset silence timer on voice activity
                
                elif event == "stop":
                    logger.info("Stop event received")
                    connection_active = False
                    break
                
                elif event == "clear":
                    logger.info("Clear event received - resetting context")
                    audio_buffer.clear()
                    silence_start = None
                    is_processing = False
                    # Reset conversation history but keep system prompt
                    history[:] = [history[0]]
                
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout - connection may be idle")
                # Send a keep-alive or check connection
                try:
                    await websocket.send_json({"event": "ping"})
                except:
                    connection_active = False
                    break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                connection_active = False
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        # Clean disconnect
        try:
            await websocket.close()
        except:
            pass
        logger.info("WebSocket connection closed")


async def send_audio_to_exotel(websocket: WebSocket, pcm_data: bytes, stream_sid: str, start_seq: int):
    """Send PCM audio to Exotel in proper chunks"""
    try:
        # Ensure PCM is in correct format
        pcm_data = ensure_linear_pcm(pcm_data)
        
        timestamp = str(int(time.time() * 1000))
        seq = start_seq
        
        # Send audio in 100ms chunks (1600 bytes at 8kHz)
        for i in range(0, len(pcm_data), CHUNK_SIZE):
            chunk = pcm_data[i:i + CHUNK_SIZE]
            
            # Pad last chunk if needed
            if len(chunk) < CHUNK_SIZE:
                chunk = chunk + b'\x00' * (CHUNK_SIZE - len(chunk))
            
            # Encode to base64
            payload = base64.b64encode(chunk).decode('utf-8')
            
            # Send media event
            await websocket.send_json({
                "event": "media",
                "sequenceNumber": str(seq),
                "streamSid": stream_sid,
                "media": {
                    "chunk": str((i // CHUNK_SIZE) + 1),
                    "timestamp": timestamp,
                    "payload": payload
                }
            })
            
            logger.debug(f"Sent chunk {(i // CHUNK_SIZE) + 1}, size: {len(chunk)}, seq: {seq}")
            seq += 1
            
            # Small delay between chunks for smooth playback
            await asyncio.sleep(SEND_INTERVAL)
        
        # Send mark event to indicate end of audio
        await websocket.send_json({
            "event": "mark",
            "sequenceNumber": str(seq),
            "streamSid": stream_sid,
            "mark": {
                "name": "end-of-audio"
            }
        })
        
        logger.info(f"Audio sent successfully, total chunks: {(len(pcm_data) // CHUNK_SIZE) + 1}")
        
    except Exception as e:
        logger.error(f"Error sending audio: {e}", exc_info=True)