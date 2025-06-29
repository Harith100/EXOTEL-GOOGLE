from flask import Flask, request, jsonify
from flask_sock import Sock
import json
import base64
import asyncio
import logging
from threading import Thread
import os
from sarvam_client import SarvamClient
from gemini_client import GeminiClient
from utils import AudioUtils
import socket
import wave
import struct
import time
import tempfile
import io
import requests


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Initialize clients
sarvam_client = SarvamClient()
gemini_client = GeminiClient()
audio_utils = AudioUtils()

# Store active connections
active_connections = {}

@app.route("/dns-test")
def dns_test():
    try:
        ip = socket.gethostbyname("api.sarvam.ai")
        return f"api.sarvam.ai resolved to: {ip}"
    except Exception as e:
        return f"DNS resolution failed: {str(e)}"

@app.route('/init', methods=['GET', 'POST'])
def init_call():
    """
    Initial endpoint that Exotel hits to get WebSocket URL
    """
    try:
        # Get call data from Exotel (can be GET params or POST JSON)
        if request.method == 'GET':
            call_data = request.args.to_dict()
        else:
            call_data = request.get_json() or {}
        
        call_sid = call_data.get('CallSid', 'unknown')
        
        logger.info(f"Initializing call: {call_sid}")
        logger.info(f"Call data: {call_data}")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request host: {request.host}")
        logger.info(f"Request URL: {request.url}")
        
        # Return WebSocket URL for media streaming
        # Get the base URL and construct WebSocket URL
        if request.is_secure or 'https' in request.url:
            ws_protocol = 'wss'
        else:
            ws_protocol = 'ws'
            
        # Use request.host to get the proper domain
        ws_url = f"{ws_protocol}://{request.host}/media"
        
        response = {
            "url": ws_url,
            "status": "initialized",
            "call_sid": call_sid
        }
        logger.info(f"response: {response}")
        
        logger.info(f"Returning WebSocket URL: {ws_url}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in init_call: {str(e)}")
        return jsonify({"error": "Initialization failed"}), 500

@sock.route('/media')
def media_handler(ws):
    """
    WebSocket handler for real-time audio streaming
    """
    connection_id = id(ws)
    active_connections[connection_id] = {
        'ws': ws,
        'audio_buffer': b'',
        'conversation_context': []
    }
    
    logger.info(f"New WebSocket connection: {connection_id}")
    
    try:
        while True:
            # Receive message from Exotel
            message = ws.receive()
            
            if not message:
                break
                
            try:
                data = json.loads(message)
                event_type = data.get('event')
                logger.info(f"Received event: {event_type}")

                if event_type == 'connected':
                    handle_connected(connection_id, data)
                    
                elif event_type == 'start':
                    handle_start(connection_id, data)
                    
                elif event_type == 'media':
                    handle_media(connection_id, data)
                    
                elif event_type == 'stop':
                    handle_stop(connection_id, data)
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Clean up connection
        if connection_id in active_connections:
            del active_connections[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")

def handle_connected(connection_id, data):
    """Handle WebSocket connected event"""
    logger.info(f"Connection {connection_id} established")
    
    # Send initial greeting
    greeting_text = "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ AI സഹായകനാണ്. എന്തെങ്കിലും സഹായം വേണോ?"
    logger.info(f"Sending greeting: {greeting_text}")
    
    # Convert to audio and send
    Thread(target=send_tts_response, args=(connection_id, greeting_text)).start()

def handle_start(connection_id, data):
    """Handle stream start event"""
    logger.info(f"Stream started for connection {connection_id}")
    active_connections[connection_id]['stream_active'] = True

def handle_media(connection_id, data):
    try:
        conn = active_connections.get(connection_id)
        if not conn:
            logger.warning(f"No active connection for ID: {connection_id}")
            return

        media = data.get("media", {})
        payload_b64 = media.get("payload", "")
        chunk_id = media.get("chunk")
        timestamp = media.get("timestamp")

        #logger.info(f"[{connection_id}] Media received — Chunk: {chunk_id}, Timestamp: {timestamp}, Payload length: {len(payload_b64)}")

        if not payload_b64:
            return

        # Decode audio from base64
        pcm_data = base64.b64decode(payload_b64)
        conn['audio_buffer'] += pcm_data
        logger.info(f"[{connection_id}] Received audio chunk, buffer size: {len(conn['audio_buffer'])} bytes")

        # --- SILENCE DETECTION START ---
        silence_threshold = 500  # Experimentally adjust if needed
        silent = True
        for i in range(0, len(pcm_data), 2):
            sample = int.from_bytes(pcm_data[i:i+2], byteorder='little', signed=True)
            if abs(sample) > silence_threshold:
                silent = False
                break

        now = time.time()
        conn.setdefault('last_voice_ts', now)
        conn.setdefault('silence_start_ts', None)

        if silent:
            if conn['silence_start_ts'] is None:
                conn['silence_start_ts'] = now
            elif now - conn['silence_start_ts'] >= 2.0:
                logger.info(f"[{connection_id}] Detected 2s silence — triggering STT")
                process_audio_chunk(connection_id)
                conn['silence_start_ts'] = None
        else:
            conn['silence_start_ts'] = None
            conn['last_voice_ts'] = now
        # --- SILENCE DETECTION END ---

        # Optional: Save raw audio for debugging
        with open(f"debug_raw_{connection_id}.pcm", "ab") as f:
            f.write(pcm_data)

    except Exception as e:
        logger.exception(f"[{connection_id}] Error in handle_media")


#def save_pcm_as_wav(pcm_data, path, sample_rate=8000):
def save_pcm_as_wav(pcm_data: bytes, sample_rate: int = 8000) -> str:
    """Convert raw PCM bytes to base64 WAV string (in-memory)."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    wav_bytes = buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')

def handle_stop(connection_id, data):
    """Handle stream stop event"""
    logger.info(f"Stream stopped for connection {connection_id}")
    if connection_id in active_connections:
        active_connections[connection_id]['stream_active'] = False
        # Process any remaining audio
        if active_connections[connection_id]['audio_buffer']:
            process_audio_chunk(connection_id, final=True)

def process_audio_chunk(connection_id):
    conn = active_connections.get(connection_id)
    if not conn:
        logger.warning(f"No connection found for ID: {connection_id}")
        return

    pcm_data = conn.get("audio_buffer", b"")
    if not pcm_data:
        logger.info(f"[{connection_id}] No audio data to process.")
        return

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            path = tmp_wav.name
            save_pcm_as_wav(pcm_data, path)
            logger.info(f"[{connection_id}] Saved audio to WAV: {path}")

            # Example STT usage
            transcription = sarvam_client.speech_to_text(path)
            logger.info(f"[{connection_id}] Transcription: {transcription}")

    except Exception as e:
        logger.exception(f"[{connection_id}] Error during STT processing.")
    finally:
        os.unlink(path)  # cleanup temp file
        conn["audio_buffer"] = b""  # clear buffer



def process_stt(connection_id, pcm_data, final=False):
    try:
        conn = active_connections.get(connection_id)
        if not conn:
            return

        wav_b64 = save_pcm_as_wav(pcm_data, sample_rate=8000)
        if not wav_b64:
            logger.error(f"[{connection_id}] Failed to convert PCM to WAV")
            return

        response = requests.post(
            "wss://api.sarvam.ai/speech-to-text/ws?language-code=ml-IN&model=saarika%3Av2.5",  # Replace with actual Sarvam STT endpoint
            json={
                "audio": {
                    "data": wav_b64,
                    "encoding": "audio/wav",
                    "sample_rate": "8000"
                }
            }
        )
        transcript = response.json().get("transcript", "")
        logger.info(f"[{connection_id}] Transcription: {transcript}")

        if transcript:
            reply_text = gemini_client.get_response(transcript)
            logger.info(f"[{connection_id}] Gemini reply: {reply_text}")

            # TTS using Sarvam
            tts_audio = sarvam_client.synthesize(reply_text)
            send_audio_response(connection_id, tts_audio)

    except Exception as e:
        logger.exception(f"[{connection_id}] Error in process_stt()")


def process_gemini_response(connection_id, user_text):
    """Get response from Gemini and convert to speech"""
    try:
        if connection_id not in active_connections:
            logger.warning(f"No active connection for ID: {connection_id}")
            return
            
        # Get conversation context
        context = active_connections[connection_id]['conversation_context']
        
        # Get response from Gemini
        response_text = gemini_client.get_response(user_text, context)
        if response_text is None:
            logger.error(f"Gemini response is None for connection {connection_id}")
            return
        
        if response_text:
            logger.info(f"Gemini response: {response_text}")
            
            # Update conversation context
            context.append({"user": user_text, "assistant": response_text})
            # Keep only last 5 exchanges to manage context size
            if len(context) > 5:
                context.pop(0)
            
            # Convert to speech and send
            send_tts_response(connection_id, response_text)
            if send_tts_response is None:
                logger.error(f"Failed to send TTS response for connection {connection_id}")
            
    except Exception as e:
        logger.error(f"Gemini processing error: {str(e)}")

def send_tts_response(connection_id, text):
    """Convert text to speech and send back"""
    try:
        if connection_id not in active_connections:
            return
            
        connection = active_connections[connection_id]
        ws = connection['ws']
        
        # Get audio from Sarvam TTS
        audio_data = sarvam_client.text_to_speech(text)
        
        if audio_data:
            logger.info(f"Audio data length: {len(audio_data)} bytes")
            # Convert audio format for Exotel (to PCM 8kHz mono)
            processed_audio = audio_utils.process_audio_for_playback(audio_data)
            logger.info(f"Processed audio length: {len(processed_audio)} bytes")
            # Split into chunks and send
            chunk_size = 3200  # 100ms chunks
            for i in range(0, len(processed_audio), chunk_size):
                chunk = processed_audio[i:i + chunk_size]
                
                # Pad chunk if necessary to maintain frame alignment
                if len(chunk) % 320 != 0:
                    padding = 320 - (len(chunk) % 320)
                    chunk += b'\x00' * padding
                
                # Encode and send
                encoded_chunk = base64.b64encode(chunk).decode('utf-8')
                logger.info(f"Sending audio chunk: {encoded_chunk[:20]}...")  # Log first 20 chars
                media_message = {
                    "event": "media",
                    "streamSid": "outbound_stream",
                    "media": {
                        "payload": encoded_chunk
                    }
                }
                
                try:
                    ws.send(json.dumps(media_message))
                except Exception as send_error:
                    logger.error(f"Error sending audio chunk: {str(send_error)}")
                    break
                    
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")

@app.route('/test-init', methods=['GET', 'POST'])
def test_init():
    """Test endpoint to debug init calls"""
    return jsonify({
        "method": request.method,
        "args": dict(request.args),
        "json": request.get_json(),
        "headers": dict(request.headers),
        "url": request.url,
        "host": request.host,
        "is_secure": request.is_secure
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "active_connections": len(active_connections)})

@app.route('/', methods=['GET'])
def home():
    """Basic home page"""
    return jsonify({
        "service": "Malayalam Voice AI Bot",
        "status": "running",
        "endpoints": {
            "init": "/init",
            "media": "wss://domain/media",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False  # Set to False in production
    )