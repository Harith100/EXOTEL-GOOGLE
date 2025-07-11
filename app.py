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

# Load knowledge base for RAG
KNOWLEDGE_BASE_FILE = os.getenv("KNOWLEDGE_BASE_FILE", "vaithiri_park_qna.json")
knowledge_base = []

def load_knowledge_base():
    """Load QnA knowledge base from JSON file"""
    global knowledge_base
    try:
        if os.path.exists(KNOWLEDGE_BASE_FILE):
            with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)
            logger.info(f"Loaded {len(knowledge_base)} QnA pairs from knowledge base")
        else:
            logger.warning(f"Knowledge base file {KNOWLEDGE_BASE_FILE} not found")
            # Create a default knowledge base
            default_kb = [
                {
                    "question": "പാർക്ക് സമയം എന്താണ്?",
                    "answer": "വൈത്തിരി പാർക്ക് രാവിലെ 9 മണി മുതൽ വൈകിട്ട് 6 മണി വരെ തുറന്നിരിക്കും.",
                    "keywords": ["സമയം", "എത്ര മണി", "തുറക്കും", "അടയ്ക്കും", "timing"]
                },
                {
                    "question": "ടിക്കറ്റ് നിരക്ക് എത്രയാണ്?",
                    "answer": "മുതിർന്നവർക്ക് ₹799, കുട്ടികൾക്ക് (90-120 സെ.മി ഉയരം) ₹599, സീനിയർ പൗരന്മാർക്ക് ₹300.",
                    "keywords": ["ടിക്കറ്റ്", "നിരക്ക്", "എൻട്രി", "ഫീസ്", "വില", "ticket", "rate", "price"]
                },
                {
                    "question": "പാർക്കിംഗ് സൗകര്യം ഉണ്ടോ?",
                    "answer": "അതെ, സൗജന്യ പാർക്കിംഗ് സൗകര്യം ലഭ്യമാണ്. കാറുകൾക്കും ബൈക്കുകൾക്കും പ്രത്യേക സ്ഥലം ഉണ്ട്.",
                    "keywords": ["പാർക്കിംഗ്", "വാഹനം", "കാർ", "ബൈക്ക്", "parking"]
                },
                {
                    "question": "ഭക്ഷണം കൊണ്ടുവരാമോ?",
                    "answer": "പുറത്തുനിന്നുള്ള ഭക്ഷണം അനുവദനീയമല്ല. പാർക്കിൽ വിവിധ തരം ഭക്ഷണശാലകളും ഫുഡ് കോർട്ടും ലഭ്യമാണ്.",
                    "keywords": ["ഭക്ഷണം", "ഫുഡ്", "കഴിക്കാൻ", "റെസ്റ്റോറന്റ്", "food"]
                },
                {
                    "question": "എത്ര റൈഡുകൾ ഉണ്ട്?",
                    "answer": "40-ലധികം റൈഡുകൾ ഉണ്ട് - അഡ്വഞ്ചർ റൈഡുകൾ, അമ്യൂസ്മെന്റ് റൈഡുകൾ, വാട്ടർ റൈഡുകൾ എന്നിവ ഉൾപ്പെടെ.",
                    "keywords": ["റൈഡ്", "റൈഡുകൾ", "അഡ്വഞ്ചർ", "rides", "attractions"]
                },
                {
                    "question": "ലൊക്കേഷൻ എവിടെയാണ്?",
                    "answer": "വൈത്തിരി പാർക്ക് വയനാട്ടിലെ വൈത്തിരിയിലാണ്. കൽപ്പറ്റയിൽ നിന്ന് 12 കി.മീ ദൂരത്തിലാണ്.",
                    "keywords": ["എവിടെ", "സ്ഥലം", "ലൊക്കേഷൻ", "വഴി", "location", "where"]
                }
            ]
            # Save default knowledge base
            with open(KNOWLEDGE_BASE_FILE, 'w', encoding='utf-8') as f:
                json.dump(default_kb, f, ensure_ascii=False, indent=2)
            knowledge_base = default_kb
            logger.info(f"Created default knowledge base with {len(default_kb)} entries")
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")

# Load knowledge base on startup
load_knowledge_base()

def retrieve_relevant_context(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve relevant QnA pairs based on the query"""
    if not knowledge_base:
        return []
    
    query_lower = query.lower()
    scored_items = []
    
    for item in knowledge_base:
        score = 0
        # Check keyword matches
        for keyword in item.get("keywords", []):
            if keyword.lower() in query_lower:
                score += 2  # Higher weight for keyword match
        
        # Check if query words appear in question
        question_lower = item["question"].lower()
        for word in query_lower.split():
            if len(word) > 2 and word in question_lower:
                score += 1
        
        if score > 0:
            scored_items.append((score, item))
    
    # Sort by score and return top_k
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored_items[:top_k]]

# Update system prompt to use RAG context
def create_system_prompt_with_context(context_items: List[Dict[str, Any]]) -> str:
    """Create system prompt with retrieved context"""
    base_prompt = """നീ വൈത്തിരി പാർക്കിന്റെ AI അസിസ്റ്റന്റാണ്. ചുരുങ്ങിയതും വ്യക്തവുമായ മറുപടികൾ മലയാളത്തിൽ നൽകുക.

പ്രധാന വിവരങ്ങൾ:
• സമയം: 9AM - 6PM
• മുതിർന്നവർ: ₹799
• കുട്ടികൾ: ₹599
• സീനിയർ: ₹300
• സ്ഥലം: വയനാട്, വൈത്തിരി"""
    
    if context_items:
        base_prompt += "\n\nപ്രസക്തമായ വിവരങ്ങൾ:\n"
        for item in context_items:
            base_prompt += f"Q: {item['question']}\nA: {item['answer']}\n\n"
    
    base_prompt += "\nമുകളിലുള്ള വിവരങ്ങൾ ഉപയോഗിച്ച് ഉപയോക്താവിന്റെ ചോദ്യത്തിന് ഉത്തരം നൽകുക. കൂടുതൽ വിവരങ്ങൾ ആവശ്യമെങ്കിൽ സെയിൽസ് ടീമുമായി ബന്ധപ്പെടാൻ നിർദേശിക്കുക."
    
    return base_prompt

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
    """Generate response using Groq LLM with RAG context"""
    if not transcript:
        return ""
    
    logger.info(f"LLM received: {transcript}")
    
    # Retrieve relevant context from knowledge base
    relevant_context = retrieve_relevant_context(transcript)
    logger.info(f"Retrieved {len(relevant_context)} relevant context items")
    
    # Create conversation-specific system prompt with context
    context_prompt = create_system_prompt_with_context(relevant_context)
    
    # Create a temporary message list with updated context
    messages = [{"role": "system", "content": context_prompt}]
    
    # Add conversation history (excluding the old system prompt)
    for msg in history[1:]:
        messages.append(msg)
    
    # Add current user message
    messages.append({"role": "user", "content": transcript})
    
    # Keep messages manageable
    if len(messages) > 20:
        messages = [messages[0]] + messages[-19:]
    
    # Retry logic for rate limiting
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            resp = groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent answers
                max_tokens=150
            )
            reply = resp.choices[0].message.content.strip()
            logger.info(f"LLM reply: {reply}")
            
            # Update history with user message and response
            history.append({"role": "user", "content": transcript})
            history.append({"role": "assistant", "content": reply})
            
            return reply
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate" in error_str.lower():
                retry_count += 1
                wait_time = min(2 ** retry_count, 10)
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
    """Convert text to PCM - with fallback to mock audio"""
    logger.info(f"Converting text to PCM via TTS: {text[:50]}...")
    
    # If in mock mode, return mock audio
    if MOCK_MODE:
        return generate_mock_pcm(text)
    
    # If Sarvam client is available, try to use it
    if sarvam_client:
        try:
            # Try different speakers if one fails
            speakers = ["manisha", "arvind", "ravi"]
            
            for speaker in speakers:
                try:
                    logger.info(f"Trying TTS with speaker: {speaker}")
                    resp = sarvam_client.text_to_speech.convert(
                        text=text,
                        target_language_code="ml-IN",
                        speaker=speaker,
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
                    
                except Exception as speaker_error:
                    logger.error(f"TTS error with speaker {speaker}: {speaker_error}")
                    continue
                    
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
    
    # Fallback to mock audio
    logger.warning("TTS failed, using mock audio")
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
    call_sid = None
    from_number = None
    audio_buffer = bytearray()
    silence_start = None
    seq_num = 1
    is_processing = False
    connection_active = True
    session_start_time = datetime.utcnow()
    
    logger.info("WebSocket connection accepted")
    
    # Extract query parameters if available
    try:
        query_params = websocket.query_params
        call_sid = query_params.get("CallSid", "unknown")
        from_number = query_params.get("From", "unknown")
        logger.info(f"Call details - CallSid: {call_sid}, From: {from_number}")
    except:
        logger.debug("No query parameters found")
    
    try:
        while connection_active:
            try:
                # Set timeout for receive to handle disconnections gracefully
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                event = msg.get("event")
                logger.debug(f"Received event: {event}")
                
                if event == "connected":
                    logger.info("Connected event received")
                    # Extract call info from connected event if available
                    if "callSid" in msg:
                        call_sid = msg.get("callSid", call_sid)
                    if "from" in msg:
                        from_number = msg.get("from", from_number)
                    continue
                
                elif event == "start":
                    stream_sid = msg.get("streamSid") or msg.get("stream_sid")
                    # Try to extract call info from start event
                    start_params = msg.get("start", {})
                    call_sid = start_params.get("callSid", call_sid)
                    from_number = start_params.get("from", from_number)
                    
                    logger.info(f"Stream started: {stream_sid}, CallSid: {call_sid}, From: {from_number}")
                    
                    # Initialize session data
                    session_data[stream_sid] = {
                        "stream_sid": stream_sid,
                        "call_sid": call_sid,
                        "from_number": from_number,
                        "start_time": session_start_time.isoformat(),
                        "conversations": []
                    }
                    
                    # Send initial greeting with fallback
                    greeting = "നമസ്കാരം! വൈത്തിരി പാർക്കിലേക്ക് സ്വാഗതം. ഞാൻ എങ്ങനെ സഹായിക്കാം?"
                    initial_pcm = text_to_pcm(greeting)
                    
                    # Add greeting to session data
                    session_data[stream_sid]["conversations"].append({
                        "role": "assistant",
                        "content": greeting,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
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
                        if transcript and stream_sid in session_data:
                            # Add to session data
                            session_data[stream_sid]["conversations"].append({
                                "role": "user",
                                "content": transcript,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                            reply = llm_respond(transcript)
                            
                            # Add reply to session data
                            session_data[stream_sid]["conversations"].append({
                                "role": "assistant",
                                "content": reply,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
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
                            if transcript and stream_sid in session_data:
                                # Add to session data
                                session_data[stream_sid]["conversations"].append({
                                    "role": "user",
                                    "content": transcript,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                
                                # Generate response
                                reply = llm_respond(transcript)
                                
                                # Add reply to session data
                                session_data[stream_sid]["conversations"].append({
                                    "role": "assistant",
                                    "content": reply,
                                    "timestamp": datetime.utcnow().isoformat()
                                })
                                
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
        # Save session data before closing
        if stream_sid and stream_sid in session_data:
            session_data[stream_sid]["end_time"] = datetime.utcnow().isoformat()
            session_data[stream_sid]["duration"] = (
                datetime.utcnow() - session_start_time
            ).total_seconds()
            
            # Log the complete session
            logger.info(f"Session ended for {from_number}")
            logger.info(f"Session data: {json.dumps(session_data[stream_sid], indent=2)}")
            
            # Save to file (optional)
            try:
                filename = f"session_{call_sid}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(session_data[stream_sid], f, indent=2)
                logger.info(f"Session saved to {filename}")
            except Exception as e:
                logger.error(f"Failed to save session: {e}")
        
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


# Add a new endpoint to retrieve session data
@app.get("/sessions/{stream_sid}")
async def get_session(stream_sid: str):
    """Retrieve session data by stream SID"""
    if stream_sid in session_data:
        return JSONResponse(content=session_data[stream_sid])
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/sessions")
async def list_sessions():
    """List all sessions"""
    sessions_summary = []
    for sid, data in session_data.items():
        sessions_summary.append({
            "stream_sid": sid,
            "call_sid": data.get("call_sid"),
            "from_number": data.get("from_number"),
            "start_time": data.get("start_time"),
            "end_time": data.get("end_time", "ongoing"),
            "conversation_count": len(data.get("conversations", []))
        })
    return JSONResponse(content=sessions_summary)
    
# Add endpoint to reload knowledge base
@app.post("/reload-knowledge-base")
async def reload_knowledge_base():
    """Reload knowledge base from file"""
    load_knowledge_base()
    return JSONResponse(content={
        "status": "success",
        "entries": len(knowledge_base),
        "file": KNOWLEDGE_BASE_FILE
    })

# Add endpoint to get current knowledge base
@app.get("/knowledge-base")
async def get_knowledge_base():
    """Get current knowledge base entries"""
    return JSONResponse(content={
        "entries": knowledge_base,
        "count": len(knowledge_base)
    })