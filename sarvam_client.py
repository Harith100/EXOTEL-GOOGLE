import requests
import base64
import json
import logging
import os
from typing import Optional, Union
import io
import wave
import tempfile
from utils import AudioUtils  # Ensure this import is available

audio_utils = AudioUtils()  # Instantiate once globally if needed

logger = logging.getLogger(__name__)

class SarvamClient:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise ValueError("SARVAM_API_KEY environment variable is required")
            logger.error("SARVAM_API_KEY not set")  
        
        self.base_url = "https://api.sarvam.ai"
        self.headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        
        # STT endpoint
        self.stt_url = f"{self.base_url}/speech-to-text"
        # TTS endpoint  
        self.tts_url = f"{self.base_url}/text-to-speech"
        
    
    
    
    def text_to_speech(self, text: str, voice: str = "meera") -> Optional[bytes]:
        """
        Convert text to speech using Sarvam TTS API and convert to Exotel-compatible PCM
        
        Args:
            text: Malayalam text to convert
            voice: Voice model to use
            
        Returns:
            PCM audio bytes (8kHz, 16-bit, mono) or None if error
        """
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided to TTS")
                return None
            
            # Prepare request payload
            payload = {
                "inputs": [text.strip()],
                "target_language_code": "ml-IN",  # Malayalam
                "speaker": voice,
                "pitch": 0,
                "pace": 1.0,
                "loudness": 1.0,
                "speech_sample_rate": 8000,  # Match Exotel format
                "enable_preprocessing": True,
                "model": "bulbul:v1"
            }
            
            # Make API request
            response = requests.post(
                self.tts_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"TTS response: {result}")
                if 'audios' in result and len(result['audios']) > 0:
                    audio_base64 = result['audios'][0]
                    wav_audio_bytes = base64.b64decode(audio_base64)

                    logger.info(f"TTS successful, audio length (WAV): {len(wav_audio_bytes)} bytes")
                    
                    # Convert to PCM 8kHz mono for Exotel
                    pcm_audio_bytes = audio_utils.process_audio_for_playback(wav_audio_bytes)

                    logger.info(f"Converted to PCM, length: {len(pcm_audio_bytes)} bytes")
                    return pcm_audio_bytes
                else:
                    logger.error("No audio data in TTS response")
                    return None
            else:
                logger.error(f"TTS API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("TTS API timeout")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"TTS API request error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"TTS processing error: {str(e)}")
            return None
    
    def get_available_voices(self) -> list:
        """
        Get list of available Malayalam voices
        
        Returns:
            List of available voice names
        """
        try:
            # This endpoint may not exist in Sarvam API
            # Return default voices for now
            return ["meera", "ravi", "arjun"]
            
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return ["meera"]  # fallback
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful
        """
        try:
            # Test with a simple TTS request
            test_text = "പരീക്ഷണം"  # "Test" in Malayalam
            result = self.text_to_speech(test_text)
            if result is None:
                logger.error("Connection test failed: No audio returned")
            return result is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
        
    def speech_to_text(self, audio_data: bytes, final: bool = False) -> Optional[str]:
    

        try:
            # Write PCM bytes to a WAV file using tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                with wave.open(temp_wav.name, 'wb') as wf:
                    logger.info(f"Writing audio data to temporary WAV file: {temp_wav.name}")
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
                    wf.setframerate(8000)
                    wf.writeframes(audio_data)

                temp_wav.seek(0)
                files = {
                    "file": ("audio.wav", open(temp_wav.name, "rb"), "audio/wav")
                }

                data = {
                    "model": "saarika:v2.5",
                    "language_code": "ml-IN",
                    "format": "wav",
                    "sample_rate": "8000",
                    "encoding": "linear16",
                    "with_timestamps": "false",
                    "enable_speaker_diarization": "false"
                }

                headers = {
                    "api-subscription-key": self.api_key
                }

                response = requests.post(
                    self.stt_url,
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=30
                )

            if response.status_code == 200:
                logger.info("STT processing successful")
                result = response.json()
                logger.info(f"STT response: {result}")
                transcript = result.get('transcript', '').strip()
                if not transcript:
                    logger.warning("STT returned empty transcript")
                return transcript if transcript else None
            else:
                logger.error(f"STT API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"STT processing error: {str(e)}")
            return None