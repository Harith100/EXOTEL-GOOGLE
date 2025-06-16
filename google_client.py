import os
import io
import base64
from google.cloud import speech
from google.cloud import texttospeech
from google.oauth2 import service_account
import logging

logger = logging.getLogger(__name__)

class GoogleCloudClient:
    def __init__(self):
        # Initialize Google Cloud clients
        self.setup_credentials()
        self.speech_client = speech.SpeechClient()
        self.tts_client = texttospeech.TextToSpeechClient()
        
        # Configure speech recognition
        self.speech_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,  # Exotel uses 8kHz
            language_code="ml-IN",  # Malayalam
            audio_channel_count=1,
            enable_automatic_punctuation=True,
        )
        
        # Configure TTS voice
        self.tts_voice = texttospeech.VoiceSelectionParams(
            language_code="ml-IN",  # Malayalam
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        # Configure TTS to output LINEAR16 WAV format
        self.tts_audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000  # Match Exotel's expected format
        )
    
    def setup_credentials(self):
        """Setup Google Cloud credentials"""
        try:
            # Try to load from environment variable (JSON string)
            if 'GOOGLE_CLOUD_CREDENTIALS_JSON' in os.environ:
                import json
                credentials_info = json.loads(os.environ['GOOGLE_CLOUD_CREDENTIALS_JSON'])
                credentials = service_account.Credentials.from_service_account_info(credentials_info)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'temp_credentials.json'
                with open('temp_credentials.json', 'w') as f:
                    json.dump(credentials_info, f)
            
            # Or load from file path
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
                logger.info("Using Google credentials from file")
            else:
                logger.warning("No Google Cloud credentials found. Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_CREDENTIALS_JSON")
                
        except Exception as e:
            logger.error(f"Error setting up Google Cloud credentials: {e}")
    
    def speech_to_text(self, wav_audio_data):
        """
        Convert speech to text using Google Cloud STT
        Expects WAV format audio data
        """
        try:
            if not wav_audio_data or len(wav_audio_data) < 1000:  # Skip very short audio
                return None
            
            # Create audio object - Google STT expects WAV format
            audio = speech.RecognitionAudio(content=wav_audio_data)
            
            # Perform recognition
            response = self.speech_client.recognize(
                config=self.speech_config,
                audio=audio
            )
            
            # Extract transcript
            if response.results:
                transcript = response.results[0].alternatives[0].transcript.strip()
                confidence = response.results[0].alternatives[0].confidence
                
                logger.info(f"STT Confidence: {confidence:.2f}")
                
                # Only return transcript if confidence is reasonable
                if confidence > 0.3:  # Adjust threshold as needed
                    return transcript
                else:
                    logger.info(f"Low confidence transcript ignored: {transcript}")
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error in speech to text: {e}")
            return None
    
    def text_to_speech(self, text):
        """
        Convert text to speech using Google Cloud TTS
        Returns WAV format audio that needs to be converted to PCM for Exotel
        """
        try:
            if not text or len(text.strip()) == 0:
                return None
            
            # Create synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Perform TTS - This returns LINEAR16 format (raw PCM)
            response = self.tts_client.synthesize_speech(
                input=synthesis_input,
                voice=self.tts_voice,
                audio_config=self.tts_audio_config
            )
            
            # Google TTS with LINEAR16 returns raw PCM data, not WAV
            # We return the raw PCM data which is ready for Exotel
            return response.audio_content
            
        except Exception as e:
            logger.error(f"Error in text to speech: {e}")
            return None
    
    def streaming_speech_to_text(self, audio_generator):
        """Streaming speech recognition (for future enhancement)"""
        try:
            config = speech.StreamingRecognitionConfig(
                config=self.speech_config,
                interim_results=True,
            )
            
            audio_requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                            for chunk in audio_generator)
            
            requests = iter([speech.StreamingRecognizeRequest(
                streaming_config=config)] + list(audio_requests))
            
            responses = self.speech_client.streaming_recognize(requests)
            
            for response in responses:
                for result in response.results:
                    if result.is_final:
                        yield result.alternatives[0].transcript
                        
        except Exception as e:
            logger.error(f"Error in streaming STT: {e}")