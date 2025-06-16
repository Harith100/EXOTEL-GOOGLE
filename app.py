from flask import Flask, request, jsonify
from flask_sock import Sock
import json
import base64
import asyncio
import threading
from google_client import GoogleCloudClient
from gemini_client import GeminiClient
from utils import AudioProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
sock = Sock(app)

# Initialize clients
google_client = GoogleCloudClient()
gemini_client = GeminiClient()
audio_processor = AudioProcessor()

class VoiceBotSession:
    def __init__(self, ws):
        self.ws = ws
        self.call_sid = None
        self.stream_sid = None
        self.is_active = True
        self.audio_buffer = bytearray()
        self.conversation_context = []
        
    def add_to_buffer(self, audio_data):
        """Add audio data to buffer and process when threshold reached"""
        self.audio_buffer.extend(audio_data)
        
        # Process when we have enough data (3200+ bytes, multiple of 320)
        if len(self.audio_buffer) >= 3200:
            # Find the largest multiple of 320 that's <= buffer size
            process_size = (len(self.audio_buffer) // 320) * 320
            chunk_to_process = bytes(self.audio_buffer[:process_size])
            self.audio_buffer = self.audio_buffer[process_size:]
            
            # Process audio chunk in separate thread
            threading.Thread(
                target=self.process_audio_chunk, 
                args=(chunk_to_process,)
            ).start()
    
    def process_audio_chunk(self, audio_chunk):
        """Process audio chunk through STT -> Gemini -> TTS pipeline"""
        try:
            logger.info(f"Processing audio chunk of size: {len(audio_chunk)}")
            
            # Convert Exotel PCM to WAV format for Google Cloud STT
            wav_audio = audio_processor.process_exotel_audio(audio_chunk)
            
            # Speech to text (expects WAV format)
            transcript = google_client.speech_to_text(wav_audio)
            if not transcript:
                logger.info("No transcript received")
                return
                
            logger.info(f"Transcript: {transcript}")
            
            # Add to conversation context
            self.conversation_context.append({"role": "user", "content": transcript})
            
            # Get response from Gemini
            response = gemini_client.get_response(transcript, self.conversation_context)
            if not response:
                logger.error("No response from Gemini")
                return
                
            logger.info(f"Gemini response: {response}")
            
            # Add bot response to context
            self.conversation_context.append({"role": "assistant", "content": response})
            
            # Keep conversation context manageable (last 10 exchanges)
            if len(self.conversation_context) > 20:
                self.conversation_context = self.conversation_context[-20:]
            
            # Text to speech (returns LINEAR16 PCM data)
            tts_pcm_data = google_client.text_to_speech(response)
            if not tts_pcm_data:
                logger.error("No audio response from TTS")
                return
            
            # Convert TTS output to Exotel PCM format
            exotel_pcm = audio_processor.convert_tts_to_exotel_format(tts_pcm_data, 'pcm')
            self.send_audio_to_exotel(exotel_pcm)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def send_audio_to_exotel(self, audio_data):
        """Send audio data back to Exotel via WebSocket"""
        try:
            # Split audio into chunks if too large
            chunk_size = 8000  # Exotel recommended chunk size
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                base64_chunk = base64.b64encode(chunk).decode('utf-8')
                
                message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {
                        "payload": base64_chunk
                    }
                }
                
                if self.is_active:
                    self.ws.send(json.dumps(message))
                    
        except Exception as e:
            logger.error(f"Error sending audio to Exotel: {e}")

@app.route('/init', methods=['POST'])
def init_call():
    """Initial endpoint that Exotel hits to get WebSocket URL"""
    try:
        # Get the base URL from the request
        base_url = request.url_root.replace('http://', 'wss://').replace('https://', 'wss://')
        websocket_url = f"{base_url}media"
        
        logger.info(f"Init call received, returning WebSocket URL: {websocket_url}")
        
        return jsonify({
            "url": websocket_url
        })
        
    except Exception as e:
        logger.error(f"Error in init endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@sock.route('/media')
def media_stream(ws):
    """WebSocket endpoint for bidirectional audio streaming"""
    logger.info("WebSocket connection established")
    session = VoiceBotSession(ws)
    
    try:
        while True:
            message = ws.receive()
            if not message:
                break
                
            try:
                data = json.loads(message)
                event = data.get('event')
                
                if event == 'connected':
                    logger.info("Stream connected")
                    
                elif event == 'start':
                    session.call_sid = data.get('start', {}).get('callSid')
                    session.stream_sid = data.get('start', {}).get('streamSid')
                    logger.info(f"Stream started - CallSid: {session.call_sid}, StreamSid: {session.stream_sid}")
                    
                    # Send welcome message
                    welcome_msg = "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ AI സഹായകനാണ്. എനിക്ക് എങ്ങനെ സഹായിക്കാൻ കഴിയും?"
                    welcome_tts = google_client.text_to_speech(welcome_msg)
                    if welcome_tts:
                        welcome_pcm = audio_processor.convert_tts_to_exotel_format(welcome_tts, 'pcm')
                        session.send_audio_to_exotel(welcome_pcm)
                    
                elif event == 'media':
                    # Decode base64 audio data
                    payload = data.get('media', {}).get('payload', '')
                    if payload:
                        audio_data = base64.b64decode(payload)
                        session.add_to_buffer(audio_data)
                
                elif event == 'stop':
                    logger.info("Stream stopped")
                    session.is_active = False
                    break
                    
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session.is_active = False
        logger.info("WebSocket connection closed")

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "Malayalam Voice AI Bot"})

@app.route('/')
def home():
    """Basic home endpoint"""
    return jsonify({
        "service": "Malayalam Voice AI Bot",
        "endpoints": {
            "init": "/init",
            "media": "/media (WebSocket)",
            "health": "/health"
        }
    })

if __name__ == '__main__':
    # For development
    app.run(host='0.0.0.0', port=5000, debug=False)