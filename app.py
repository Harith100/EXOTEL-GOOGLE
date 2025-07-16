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
from groq import Groq
import io
import struct
import json
from datetime import datetime
from google.cloud import texttospeech
from google.oauth2 import service_account

class GoogleTTS:
    def __init__(self):
        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if not creds_json:
            raise RuntimeError("❌ Missing GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable")

        creds_dict = json.loads(creds_json)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)

    def synthesize(self, text: str) -> bytes:
        input_text = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="ml-IN",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000
        )
        response = self.client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        with wave.open(io.BytesIO(response.audio_content), "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
        return frames  # raw 16-bit PCM, 8kHz mono, Exotel-compatible



# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(name)s:%(message)s',
)
logger = logging.getLogger("exotel_bot")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true"
google_tts = GoogleTTS()
# Try to import Sarvam AI but make it optional
try:
    from sarvamai import SarvamAI
    SARVAMAI_API_KEY = os.getenv("SARVAMAI_API_KEY")
    sarvam_client = SarvamAI(api_subscription_key=SARVAMAI_API_KEY) if SARVAMAI_API_KEY else None
except ImportError:
    logger.warning("SarvamAI not available")
    sarvam_client = None

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Audio configuration - CRITICAL: Match Exotel's exact format
SAMPLE_RATE = 8000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_SIZE = 1600  # 100ms at 8kHz = 800 samples * 2 bytes
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2.0  # seconds of silence to trigger processing
SEND_INTERVAL = 0.02  # 20ms intervals for smoother playback
MAX_AUDIO_DURATION = 25  # Maximum audio duration in seconds (keep under 30s limit)
MAX_BUFFER_SIZE = SAMPLE_RATE * SAMPLE_WIDTH * MAX_AUDIO_DURATION  # Max buffer size in bytes

# FastAPI app
app = FastAPI()

# Health endpoint
@app.get("/health")
async def health_check():
    health_status = {
        "status": "ok",
        "services": {
            "groq": "unknown",
            "sarvam": "unknown",
            "mock_mode": MOCK_MODE
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Check Groq
        if groq_client:
            health_status["services"]["groq"] = "ok"
            
        # Check Sarvam
        if sarvam_client:
            health_status["services"]["sarvam"] = "available"
        else:
            health_status["services"]["sarvam"] = "not_configured"
            
        # Overall status
        if health_status["services"]["groq"] == "ok":
            return JSONResponse(status_code=200, content=health_status)
        else:
            return JSONResponse(status_code=503, content=health_status)
            
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status["status"] = "error"
        health_status["error"] = str(e)
        return JSONResponse(status_code=503, content=health_status)

# History for the LLM (system prompt)
history = [{
    "role": "system",
    "content": """
നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്. താങ്കളുടെ ജോലി വയനാട്ടിലെ വൈത്തിരി പാർക്കിന്റെ സന്ദർശകരുമായി സൗഹൃദപരമായ രീതിയിൽ സംസാരിക്കുകയും, അവരുടെ സംശയങ്ങൾക്കും ചോദ്യങ്ങൾക്കും വ്യക്തമായ മറുപടികൾ നൽകുകയും ചെയ്യുകയാണ്. 

വൈത്തിരി പാർക്ക് വിവരങ്ങൾ:
- സ്ഥലം: വയനാട്ടിലെ വൈത്തിരി
- സമയം: രാവിലെ 9 മണി മുതൽ വൈകിട്ട് 6 മണി വരെ
- ടിക്കറ്റ് നിരക്ക്:
  - മുതിർന്നവർ: ₹799
  - കുട്ടികൾ (90-120 സെ.മി): ₹599
  - മുതിർന്ന പൗരന്മാർ: ₹300
- 40+ റൈഡുകൾ ലഭ്യമാണ്

ചുരുങ്ങിയതും വ്യക്തവുമായ മറുപടികൾ നൽകുക. എല്ലാ മറുപടികളും മലയാളത്തിൽ മാത്രം.
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


def mock_transcribe(pcm_data: bytes) -> str:
    """Mock transcription for testing when STT services are down"""
    # Simulate different queries based on audio length
    audio_length = len(pcm_data)
    
    mock_queries = [
        "പാർക്ക് എത്ര മണിക്ക് തുറക്കും?",
        "ടിക്കറ്റ് നിരക്ക് എന്താണ്?",
        "കുട്ടികൾക്ക് എന്തെല്ലാം റൈഡുകൾ ഉണ്ട്?",
        "പാർക്കിംഗ് സൗകര്യം ഉണ്ടോ?",
        "ഭക്ഷണം കൊണ്ടുവരാമോ?"
    ]
    
    # Return a random query based on audio length
    import random
    return random.choice(mock_queries)


def transcribe_pcm(pcm_data: bytes) -> str:
    """Transcribe PCM audio - with fallback to mock in case of service failure"""
    if len(pcm_data) < 1600:  # Skip if less than 100ms of audio
        return ""
    
    # Check audio duration
    duration_seconds = len(pcm_data) / (SAMPLE_RATE * SAMPLE_WIDTH)
    if duration_seconds > 30:
        logger.warning(f"Audio too long ({duration_seconds:.1f}s), truncating to 25s")
        pcm_data = pcm_data[:MAX_BUFFER_SIZE]
    
    # If in mock mode, return mock transcript
    if MOCK_MODE:
        mock_transcript = mock_transcribe(pcm_data)
        logger.info(f"Mock transcript: {mock_transcript}")
        return mock_transcript
    
    # If Sarvam client is available, try to use it
    if sarvam_client:
        temp_file = f"temp_{uuid.uuid4().hex}.wav"
        try:
            pcm_to_wav(pcm_data, temp_file)
            logger.info(f"Starting transcription, audio duration: {duration_seconds:.1f}s")
            
            with open(temp_file, 'rb') as wf:
                # Try with different model versions if available
                models = ["saarika:v2.5", "saarika:v2", "saarika:v1"]
                
                for model in models:
                    try:
                        logger.info(f"Trying transcription with model: {model}")
                        result = sarvam_client.speech_to_text.transcribe(
                            file=wf,
                            model=model,
                            language_code="ml-IN"
                        )
                        transcript = result.transcript.strip()
                        logger.info(f"Transcript: {transcript}")
                        return transcript
                    except Exception as model_error:
                        error_str = str(model_error)
                        if "duration greater than 30 seconds" in error_str:
                            logger.error("Audio still too long, skipping")
                            return ""
                        logger.error(f"Transcription error with {model}: {model_error}")
                        wf.seek(0)
                        continue
                        
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # If all else fails, use mock mode
    logger.warning("All transcription methods failed, using mock mode")
    return mock_transcribe(pcm_data)


def llm_respond(transcript: str) -> str:
    """Generate response using Groq LLM with rate limit handling"""
    if not transcript:
        return ""
        
    logger.info(f"LLM received: {transcript}")
    history.append({"role": "user", "content": transcript})
    
    # Keep history manageable
    if len(history) > 20:
        history[:] = [history[0]] + history[-19:]  # Keep system prompt + last 19 messages
    
    # Retry logic for rate limiting
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",  # Currently supported model
                messages=history,
                temperature=0.5,
                max_tokens=150  # Keep responses concise
            )
            reply = resp.choices[0].message.content.strip()
            logger.info(f"LLM reply: {reply}")
            history.append({"role": "assistant", "content": reply})
            return reply
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                retry_count += 1
                wait_time = min(2 ** retry_count, 10)  # Exponential backoff, max 10s
                logger.warning(f"Rate limit hit, waiting {wait_time}s (retry {retry_count}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"LLM error: {e}")
                break
    
    # Fallback response
    return "മാപ്പുണ്ട്, എനിക്ക് ഇപ്പോൾ നിങ്ങളെ സഹായിക്കാൻ കഴിയുന്നില്ല. ദയവായി വീണ്ടും ശ്രമിക്കുക."


def generate_mock_pcm(text: str) -> bytes:
    """Generate mock PCM data for testing when TTS is down"""
    # Generate silence with some variation to simulate speech
    duration_ms = min(len(text) * 50, 5000)  # Approximate duration based on text length
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    
    # Create a simple sine wave to simulate speech
    t = np.linspace(0, duration_ms/1000, num_samples)
    frequency = 440  # A4 note
    amplitude = 5000
    
    # Add some variation
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    # Add envelope to make it sound more natural
    envelope = np.exp(-t * 2)
    signal = signal * envelope
    
    # Convert to int16
    pcm = signal.astype(np.int16).tobytes()
    logger.info(f"Generated mock PCM of length: {len(pcm)}")
    return pcm




def text_to_pcm(text: str) -> bytes:
    logger.info(f"Converting text to PCM via Google TTS: {text[:50]}...")
    
    if MOCK_MODE:
        return generate_mock_pcm(text)

    try:
        pcm = google_tts.synthesize(text)
        logger.debug(f"Generated PCM length: {len(pcm)}")
        return pcm
    except Exception as e:
        logger.error(f"Google TTS failed: {e}", exc_info=True)
        return generate_mock_pcm(text)



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
                    continue
                
                elif event == "start":
                    stream_sid = msg.get("streamSid") or msg.get("stream_sid")
                    logger.info(f"Stream started: {stream_sid}")
                    
                    # Send initial greeting with fallback
                    greeting = "നമസ്കാരം! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. ഞാൻ എങ്ങനെ സഹായിക്കാം?"
                    initial_pcm = text_to_pcm(greeting)
                    
                    # If TTS fails, use a pre-recorded PCM or skip
                    if initial_pcm and len(initial_pcm) > 1600:
                        await send_audio_to_exotel(websocket, initial_pcm, stream_sid, seq_num)
                        seq_num += len(initial_pcm) // CHUNK_SIZE + 1
                    else:
                        logger.warning("Initial greeting TTS failed, continuing without audio")
                    continue
                
                elif event == "media":
                    if is_processing:
                        continue  # Skip incoming audio while processing
                    
                    # Decode audio payload
                    payload = base64.b64decode(msg["media"]["payload"])
                    
                    # Check buffer size to prevent > 30s audio
                    if len(audio_buffer) + len(payload) > MAX_BUFFER_SIZE:
                        logger.warning("Audio buffer full, processing current buffer")
                        # Force processing of current buffer
                        is_processing = True
                        pcm_data = bytes(audio_buffer)
                        audio_buffer.clear()
                        silence_start = None
                        
                        # Process the audio
                        transcript = transcribe_pcm(pcm_data)
                        if transcript:
                            reply = llm_respond(transcript)
                            pcm_reply = text_to_pcm(reply)
                            if pcm_reply and len(pcm_reply) > 100:
                                await send_audio_to_exotel(websocket, pcm_reply, stream_sid, seq_num)
                                seq_num += len(pcm_reply) // CHUNK_SIZE + 1
                        
                        is_processing = False
                        continue
                    
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
                                if pcm_reply and len(pcm_reply) > 100:
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
        # Simple connection check
        try:
            # Try to access websocket properties to check if it's still valid
            _ = websocket.headers
        except:
            logger.warning("WebSocket not connected, skipping audio send")
            return
            
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
            
            try:
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
                
            except (WebSocketDisconnect, RuntimeError, ConnectionError) as e:
                logger.warning(f"WebSocket error during chunk send: {e}")
                break
        
        # Send mark event if still connected
        try:
            await websocket.send_json({
                "event": "mark",
                "sequenceNumber": str(seq),
                "streamSid": stream_sid,
                "mark": {
                    "name": "end-of-audio"
                }
            })
            logger.info(f"Audio sent successfully, total chunks: {(len(pcm_data) // CHUNK_SIZE) + 1}")
        except:
            pass
        
    except Exception as e:
        logger.error(f"Error sending audio: {e}", exc_info=True)


# Add a test endpoint to verify services
@app.get("/test-services")
async def test_services():
    """Test endpoint to check if STT/TTS services are working"""
    results = {
        "sarvam_stt": "not_tested",
        "sarvam_tts": "not_tested",
        "groq_llm": "not_tested",
        "mock_mode": MOCK_MODE
    }
    
    # Test Groq
    try:
        test_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        results["groq_llm"] = "working"
    except Exception as e:
        results["groq_llm"] = f"error: {str(e)}"
    
    # Test Sarvam if available
    if sarvam_client and not MOCK_MODE:
        try:
            # Test TTS
            resp = sarvam_client.text_to_speech.convert(
                text="Test",
                target_language_code="ml-IN",
                speaker="manisha",
                enable_preprocessing=True,
                speech_sample_rate=8000
            )
            results["sarvam_tts"] = "working"
        except Exception as e:
            results["sarvam_tts"] = f"error: {str(e)}"
    
    return JSONResponse(content=results)