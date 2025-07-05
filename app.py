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
import uvicorn
from starlette.websockets import WebSocketState
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO to reduce noise
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
SILENCE_DURATION = 1.5  # Reduced from 2.0 seconds for better responsiveness

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
    if not audio_bytes:
        return True
    try:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(audio_np) == 0:
            return True
        rms = np.sqrt(np.mean(audio_np.astype(np.float64) ** 2))
        return rms < SILENCE_THRESHOLD
    except Exception as e:
        logger.error(f"Error in silence detection: {e}")
        return True

def pcm_to_wav(pcm_data: bytes, wav_path: str) -> None:
    try:
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_data)
    except Exception as e:
        logger.error(f"Error creating WAV file: {e}")
        raise

def transcribe_pcm(pcm_data: bytes) -> str:
    if len(pcm_data) < 1600:  # Less than 50ms of audio
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
        logger.error(f"Transcription failed: {e}")
        return ""
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def llm_respond(transcript: str) -> str:
    try:
        logger.info(f"LLM received: {transcript}")
        
        # Create a copy of history for this request
        current_history = history.copy()
        current_history.append({"role": "user", "content": transcript})
        
        # Keep only last 10 messages to prevent context overflow
        if len(current_history) > 11:  # 1 system + 10 conversation messages
            current_history = current_history[:1] + current_history[-10:]
        
        resp = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=current_history,
            temperature=0.5,
            max_tokens=150  # Limit response length for voice calls
        )
        reply = resp.choices[0].message.content.strip()
        logger.info(f"LLM reply: {reply}")
        
        # Update global history
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})
        
        # Trim global history
        if len(history) > 21:  # 1 system + 20 conversation messages
            history[:] = history[:1] + history[-20:]
            
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "ക്ഷമിക്കണം, എനിക്ക് മനസ്സിലായില്ല. വീണ്ടും പറയാമോ?"

def text_to_pcm(text: str) -> bytes:
    try:
        logger.info("Converting text to PCM via TTS")
        resp = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code="ml-IN",
            speaker="manisha",
            enable_preprocessing=True,
            speech_sample_rate=SAMPLE_RATE
        )
        pcm = b"".join(base64.b64decode(chunk) for chunk in resp.audios)
        logger.info(f"Generated PCM length: {len(pcm)}")
        return pcm
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise

def chunk_pcm(pcm_data: bytes, size: int = CHUNK_MIN_SIZE):
    for i in range(0, len(pcm_data), size):
        chunk = pcm_data[i:i+size]
        trim = len(chunk) % CHUNK_ALIGNMENT
        if trim:
            chunk = chunk[:-trim]
        if chunk:
            yield chunk

# Helper function to safely send WebSocket messages
async def safe_send_json(websocket: WebSocket, data: dict) -> bool:
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(data)
            return True
        else:
            logger.warning("WebSocket not connected, cannot send message")
            return False
    except Exception as e:
        logger.error(f"Error sending WebSocket message: {e}")
        return False

# -- WebSocket endpoint --
@app.websocket("/ws")
async def ws_exotel(websocket: WebSocket):
    await websocket.accept()
    stream_sid = None
    audio_buffer = deque()
    silence_start = None
    seq_num = 1
    processing = False

    logger.info("WebSocket connection accepted")

    try:
        while True:
            try:
                # Check if websocket is still connected
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.warning("WebSocket disconnected")
                    break
                
                # Receive message with timeout
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                event = msg.get("event")
                
                if event == "connected":
                    logger.info("Connected event received")
                    continue

                elif event == "start":
                    stream_sid = msg.get("stream_sid")
                    logger.info(f"Stream started: {stream_sid}")
                    continue

                elif event == "media":
                    if processing:
                        continue  # Skip new audio while processing
                    
                    payload = base64.b64decode(msg["media"]["payload"])
                    audio_buffer.append(payload)

                    if is_silence(payload):
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start >= SILENCE_DURATION:
                            if len(audio_buffer) > 5:  # Ensure we have enough audio
                                processing = True
                                pcm_data = b"".join(audio_buffer)
                                audio_buffer.clear()
                                silence_start = None
                                
                                # Process audio in background
                                asyncio.create_task(process_audio(websocket, pcm_data, stream_sid, seq_num))
                                seq_num += 50  # Reserve sequence numbers
                            else:
                                silence_start = None
                    else:
                        silence_start = None

                elif event == "stop":
                    logger.info("Stop event received; closing connection")
                    break

                elif event == "mark":
                    # Handle mark events if needed
                    mark_name = msg.get("mark", {}).get("name")
                    if mark_name == "end-of-reply":
                        processing = False
                    logger.debug(f"Received mark event: {mark_name}")

            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                break
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
            logger.info("WebSocket closed gracefully")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")

async def process_audio(websocket: WebSocket, pcm_data: bytes, stream_sid: str, base_seq_num: int):
    """Process audio data and send response"""
    try:
        logger.info("Processing buffered audio after silence")
        
        # Transcribe audio
        transcript = await asyncio.to_thread(transcribe_pcm, pcm_data)
        if not transcript:
            return

        # Get LLM response
        reply = await asyncio.to_thread(llm_respond, transcript)

        # Send keep-alive mark
        await safe_send_json(websocket, {
            "event": "mark",
            "sequence_number": base_seq_num,
            "stream_sid": stream_sid,
            "mark": {"name": "thinking"}
        })

        # Generate TTS
        try:
            pcm_reply = await asyncio.to_thread(text_to_pcm, reply)
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            pcm_reply = await asyncio.to_thread(text_to_pcm, "ക്ഷമിക്കണം, തകരാറുണ്ടായി.")

        # Send audio chunks
        timestamp = str(int(time.time() * 1000))
        chunk_idx = 1
        seq_num = base_seq_num + 1

        for chunk in chunk_pcm(pcm_reply):
            success = await safe_send_json(websocket, {
                "event": "media",
                "sequence_number": seq_num,
                "stream_sid": stream_sid,
                "media": {
                    "chunk": chunk_idx,
                    "timestamp": timestamp,
                    "payload": base64.b64encode(chunk).decode()
                }
            })
            
            if not success:
                logger.warning("Failed to send media chunk, stopping")
                break
                
            seq_num += 1
            chunk_idx += 1
            await asyncio.sleep(0.01)  # Reduced delay for better responsiveness

        # Send end-of-reply mark
        await safe_send_json(websocket, {
            "event": "mark",
            "sequence_number": seq_num,
            "stream_sid": stream_sid,
            "mark": {"name": "end-of-reply"}
        })

    except Exception as e:
        logger.error(f"Error in process_audio: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000, log_level="info")