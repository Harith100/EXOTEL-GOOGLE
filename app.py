import os
import time
import base64
import uuid
import wave
import io
import asyncio
import numpy as np
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import JSONResponse
from collections import deque
from sarvamai import SarvamAI
from groq import Groq
import tempfile

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
CHUNK_MIN_SIZE = 3200  # 100ms (8000 Hz * 0.1s * 2 bytes/sample)
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
നീ മലയാളത്തിൽ സംസാരിക്കുന്ന ഒരു വോയ്സ് കോളിന്റെ അസിസ്റ്റന്റാണ്. വിനീതമായി സംസാരിക്കണം, വ്യക്തതയോടെ മറുപടി പറയണം, സന്ദർശകർക്ക് വൈത്തിരി പാർക്കിനെ കുറിച്ച് സഹായിക്കണം.
"""
}]

# ---------------- UTILS ----------------

def is_silence(audio_bytes: bytes) -> bool:
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio_np) == 0:
            return True
        
        # Calculate RMS with overflow protection
        mean_square = np.mean(audio_np.astype(np.float64) ** 2)
        rms = np.sqrt(mean_square)
        
        # Handle NaN or invalid values
        if np.isnan(rms) or np.isinf(rms):
            logger.warning(f"Invalid RMS value: {rms}")
            return False  # Don't treat as silence if we can't calculate properly
        
        logger.debug(f"RMS: {rms}")
        return rms < SILENCE_THRESHOLD
    except Exception as e:
        logger.error(f"Error calculating RMS: {e}")
        return False

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
        if os.path.exists(temp_file):
            os.remove(temp_file)

def llm_respond(transcript: str) -> str:
    logger.info(f"LLM received: {transcript}")
    history.append({"role": "user", "content": transcript})
    if len(history) > 20:
        history[:] = history[-20:]
    try:
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=history,
            temperature=0.5
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ക്ഷമിക്കണം, എനിക്ക് ഇപ്പോൾ മറുപടി നൽകാൻ കഴിയുന്നില്ല."

def text_to_pcm_wav(text: str) -> bytes:
    """Convert text to WAV using Sarvam AI TTS"""
    logger.info("Converting text to WAV via TTS")
    try:
        # Call Sarvam TTS API
        resp = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code="ml-IN",
            speaker="manisha",  # Female voice options: anushka, manisha, vidya, arya
            enable_preprocessing=True,
            speech_sample_rate=SAMPLE_RATE,  # 8000 Hz for Exotel
            pace=1.0,  # Normal speed
            pitch=0.0,  # Normal pitch
            loudness=1.0  # Normal volume
        )
        
        # Debug logging
        logger.debug(f"TTS response audios type: {type(resp.audios)}")
        logger.debug(f"TTS response audios length: {len(resp.audios)}")
        
        # Sarvam returns a list with base64 encoded WAV data
        # Usually it's a single element in the list
        if resp.audios and len(resp.audios) > 0:
            # Decode the base64 WAV data
            wav_bytes = base64.b64decode(resp.audios[0])
            logger.info(f"TTS returned {len(wav_bytes)} bytes of WAV data")
            
            # Verify it's a valid WAV file by checking the header
            if wav_bytes[:4] == b'RIFF' and wav_bytes[8:12] == b'WAVE':
                logger.debug("Valid WAV file header detected")
            else:
                logger.warning("WAV file header not found")
            
            return wav_bytes
        else:
            raise ValueError("No audio data returned from TTS")
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise

def wav_to_pcm_chunks(wav_bytes: bytes, chunk_size: int = CHUNK_MIN_SIZE):
    """Extract raw PCM from WAV and yield aligned chunks"""
    logger.info("Converting WAV to raw PCM chunks")
    
    try:
        # Use BytesIO to read WAV from memory
        wav_buffer = io.BytesIO(wav_bytes)
        
        with wave.open(wav_buffer, 'rb') as wf:
            # Verify WAV parameters match Exotel requirements
            if wf.getframerate() != SAMPLE_RATE:
                logger.warning(f"Sample rate mismatch: {wf.getframerate()} != {SAMPLE_RATE}")
            if wf.getsampwidth() != SAMPLE_WIDTH:
                logger.warning(f"Sample width mismatch: {wf.getsampwidth()} != {SAMPLE_WIDTH}")
            if wf.getnchannels() != CHANNELS:
                logger.warning(f"Channels mismatch: {wf.getnchannels()} != {CHANNELS}")
            
            # Extract raw PCM data (without WAV headers)
            pcm_bytes = wf.readframes(wf.getnframes())
            logger.info(f"Extracted {len(pcm_bytes)} bytes of PCM data")
        
        # Yield chunks of PCM data aligned to CHUNK_ALIGNMENT (320 bytes)
        total_chunks = 0
        for i in range(0, len(pcm_bytes), chunk_size):
            chunk = pcm_bytes[i:i + chunk_size]
            
            # Pad the last chunk if needed to maintain alignment
            if len(chunk) < chunk_size and len(chunk) % CHUNK_ALIGNMENT != 0:
                padding_needed = CHUNK_ALIGNMENT - (len(chunk) % CHUNK_ALIGNMENT)
                chunk += b'\x00' * padding_needed
                logger.debug(f"Padded chunk with {padding_needed} bytes")
            
            total_chunks += 1
            yield chunk
        
        logger.info(f"Generated {total_chunks} PCM chunks")
        
    except Exception as e:
        logger.error(f"Error converting WAV to PCM: {e}")
        raise

def text_to_exotel_pcm_chunks(text: str):
    """Convert text to PCM chunks ready for Exotel streaming"""
    try:
        # Get WAV bytes from TTS
        wav_bytes = text_to_pcm_wav(text)
        # Convert WAV to PCM chunks
        return list(wav_to_pcm_chunks(wav_bytes))
    except Exception as e:
        logger.error(f"Error in text_to_exotel_pcm_chunks: {e}")
        return []

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

                # Send welcome message
                welcome_text = "വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം! ഞാൻ നിങ്ങളെ സഹായിക്കാനാണ് ഇവിടെ."
                try:
                    chunks = text_to_exotel_pcm_chunks(welcome_text)
                    logger.info(f"Sending {len(chunks)} welcome chunks")
                    
                    for i, chunk in enumerate(chunks):
                        # Verify chunk is valid before sending
                        if len(chunk) == 0:
                            logger.warning(f"Empty chunk {i+1}, skipping")
                            continue
                            
                        await websocket.send_json({
                            "event": "media",
                            "sequence_number": seq_num,
                            "stream_sid": stream_sid,
                            "media": {
                                "chunk": seq_num,
                                "timestamp": str(int(time.time() * 1000)),
                                "payload": base64.b64encode(chunk).decode()
                            }
                        })
                        logger.debug(f"Sent welcome chunk {seq_num} ({i+1}/{len(chunks)}), size={len(chunk)}")
                        seq_num += 1
                        # Small delay between chunks to prevent overwhelming
                        await asyncio.sleep(0.01)
                    
                    # Send a mark event after welcome message
                    await websocket.send_json({
                        "event": "mark",
                        "sequence_number": seq_num,
                        "stream_sid": stream_sid,
                        "mark": {"name": "welcome-complete"}
                    })
                    logger.info(f"Welcome message complete, sent mark event")
                    seq_num += 1
                    
                except Exception as e:
                    logger.error(f"Error sending welcome message: {e}", exc_info=True)
                continue

            if event == "media":
                payload = base64.b64decode(msg["media"]["payload"])
                audio_buffer.append(payload)
                logger.debug(f"Buffered media chunk, size={len(payload)}")

                # Check if audio contains speech or is silence
                if is_silence(payload):
                    if silence_start is None:
                        silence_start = time.time()
                        logger.debug("Silence detected, starting timer")
                    elif time.time() - silence_start >= SILENCE_DURATION:
                        # Process buffered audio only if we have enough data
                        if len(audio_buffer) > 10:  # At least 10 chunks (1 second)
                            pcm_data = b"".join(audio_buffer)
                            audio_buffer.clear()
                            silence_start = None
                            logger.info(f"Processing {len(pcm_data)} bytes of buffered audio after silence")

                            # Transcribe
                            transcript = transcribe_pcm(pcm_data)
                            if not transcript:
                                logger.info("No transcript, continuing...")
                                continue

                            # Get LLM response
                            reply = llm_respond(transcript)
                            
                            # Convert reply to PCM and send
                            try:
                                chunks = text_to_exotel_pcm_chunks(reply)
                                for chunk in chunks:
                                    await websocket.send_json({
                                        "event": "media",
                                        "sequence_number": seq_num,
                                        "stream_sid": stream_sid,
                                        "media": {
                                            "chunk": seq_num,
                                            "timestamp": str(int(time.time() * 1000)),
                                            "payload": base64.b64encode(chunk).decode()
                                        }
                                    })
                                    logger.debug(f"Sent reply chunk {seq_num}, size={len(chunk)}")
                                    seq_num += 1
                                    await asyncio.sleep(0.01)

                                # Send mark event to indicate end of response
                                await websocket.send_json({
                                    "event": "mark",
                                    "sequence_number": seq_num,
                                    "stream_sid": stream_sid,
                                    "mark": {"name": "end-of-reply"}
                                })
                                logger.info(f"Sent mark event, seq={seq_num}")
                                seq_num += 1
                            except Exception as e:
                                logger.error(f"Error sending reply: {e}")
                        else:
                            # Reset silence timer if we don't have enough audio
                            silence_start = time.time()
                else:
                    # Reset silence timer if voice detected
                    if silence_start is not None:
                        logger.debug("Voice detected, resetting silence timer")
                    silence_start = None

            if event == "stop":
                logger.info("Stop event received; closing connection")
                break

            if event == "clear":
                logger.info("Clear event received; resetting context")
                audio_buffer.clear()
                silence_start = None
                # Optionally reset conversation history
                # history[:] = [history[0]]  # Keep only system message

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        if websocket.client_state.name == "CONNECTED":
            await websocket.close()
            logger.info("WebSocket closed safely")
        else:
            logger.info("WebSocket already closed")