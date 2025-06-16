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

        logger.info(f"[{connection_id}] Media received — Chunk: {chunk_id}, Timestamp: {timestamp}, Payload length: {len(payload_b64)}")

        if not payload_b64:
            logger.warning(f"[{connection_id}] Empty payload received")
            return

        # Base64 decode
        try:
            pcm_data = base64.b64decode(payload_b64)
            logger.debug(f"[{connection_id}] PCM decoded — Length: {len(pcm_data)} bytes")
        except Exception as e:
            logger.exception(f"[{connection_id}] Base64 decode failed")
            return

        # Append to buffer
        conn['audio_buffer'] += pcm_data
        logger.debug(f"[{connection_id}] Buffer size after append: {len(conn['audio_buffer'])} bytes")

        # Optional: Save raw PCM for playback
        with open(f"debug_raw_{connection_id}.pcm", "ab") as f:
            f.write(pcm_data)

        # Optional: Save to WAV every 2 seconds (or when needed)
        if len(conn['audio_buffer']) >= 32000:
            wav_path = f"debug_audio_{connection_id}.wav"
            save_pcm_as_wav(conn['audio_buffer'], wav_path)
            logger.info(f"[{connection_id}] Saved WAV file for STT: {wav_path}")
            conn['audio_buffer'] = b''

    except Exception as e:
        logger.exception(f"[{connection_id}] Error in handle_media")

def save_pcm_as_wav(pcm_data, path, sample_rate=8000):
    """Helper to save raw PCM to WAV"""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

def handle_stop(connection_id, data):
    """Handle stream stop event"""
    logger.info(f"Stream stopped for connection {connection_id}")
    if connection_id in active_connections:
        active_connections[connection_id]['stream_active'] = False
        # Process any remaining audio
        if active_connections[connection_id]['audio_buffer']:
            process_audio_chunk(connection_id, final=True)

def process_audio_chunk(connection_id, final=False):
    """Process accumulated audio buffer"""
    if connection_id not in active_connections:
        return
        
    connection = active_connections[connection_id]
    audio_buffer = connection['audio_buffer']
    
    if not audio_buffer:
        return
        
    try:
        # Ensure chunk size is multiple of 320 (frame size for 8kHz)
        chunk_size = (len(audio_buffer) // 320) * 320
        if chunk_size == 0:
            return
            
        audio_chunk = audio_buffer[:chunk_size]
        connection['audio_buffer'] = audio_buffer[chunk_size:]
        
        # Convert audio format for Sarvam (PCM 8kHz to required format)
        processed_audio = audio_utils.process_audio_for_stt(audio_chunk)
        
        # Send to STT in separate thread to avoid blocking
        Thread(target=process_stt, args=(connection_id, processed_audio, final)).start()
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")

def process_stt(connection_id, audio_data, final=False):
    """Process Speech-to-Text"""
    try:
        # Send audio to Sarvam STT
        transcript = sarvam_client.speech_to_text(audio_data, final=final)
        
        if transcript and transcript.strip():
            logger.info(f"Transcript: {transcript}")
            
            # Send to Gemini for response
            Thread(target=process_gemini_response, args=(connection_id, transcript)).start()
            
    except Exception as e:
        logger.error(f"STT error: {str(e)}")

def process_gemini_response(connection_id, user_text):
    """Get response from Gemini and convert to speech"""
    try:
        if connection_id not in active_connections:
            return
            
        # Get conversation context
        context = active_connections[connection_id]['conversation_context']
        
        # Get response from Gemini
        response_text = gemini_client.get_response(user_text, context)
        
        if response_text:
            logger.info(f"Gemini response: {response_text}")
            
            # Update conversation context
            context.append({"user": user_text, "assistant": response_text})
            # Keep only last 5 exchanges to manage context size
            if len(context) > 5:
                context.pop(0)
            
            # Convert to speech and send
            send_tts_response(connection_id, response_text)
            
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
            # Convert audio format for Exotel (to PCM 8kHz mono)
            processed_audio = audio_utils.process_audio_for_playback(audio_data)
            
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