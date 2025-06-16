import base64
import logging
import numpy as np
from typing import Optional
import io
import wave
import struct


logger = logging.getLogger(__name__)

class AudioUtils:
    """Utility class for audio processing and format conversion"""
    
    def __init__(self):
        # Exotel audio format specifications
        self.exotel_sample_rate = 8000  # 8kHz
        self.exotel_channels = 1        # Mono
        self.exotel_sample_width = 2    # 16-bit
        self.exotel_format = 'slin'     # Signed linear PCM
        
        # Frame size for 8kHz, 16-bit, mono (20ms frame = 160 samples = 320 bytes)
        self.frame_size = 320
        self.min_chunk_size = 3200      # 100ms minimum
        self.max_chunk_size = 100000    # 100KB maximum
    
    def process_audio_for_stt(self, audio_data: bytes) -> bytes:
        """
        Process audio data for STT (Exotel format to Sarvam format)
        
        Args:
            audio_data: Raw PCM audio bytes from Exotel (8kHz, 16-bit, mono)
            
        Returns:
            Processed audio bytes suitable for Sarvam STT
        """
        try:
            if not audio_data:
                return b''
                
            # Exotel sends 8kHz, 16-bit, mono PCM (little-endian)
            # Sarvam expects the same format, so minimal processing needed
            
            # Ensure proper frame alignment
            if len(audio_data) % self.frame_size != 0:
                # Pad with silence to align to frame boundary
                padding_needed = self.frame_size - (len(audio_data) % self.frame_size)
                audio_data += b'\x00' * padding_needed
                
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing audio for STT: {str(e)}")
            return b''
    
    def process_audio_for_playback(self, audio_data: bytes) -> bytes:
        """
        Process audio data for playback (Sarvam TTS output to Exotel format)
        
        Args:
            audio_data: Audio bytes from Sarvam TTS
            
        Returns:
            Processed audio bytes suitable for Exotel playback
        """
        try:
            if not audio_data:
                return b''
            
            # Try to detect if it's WAV format
            if self._is_wav_format(audio_data):
                # Extract PCM data from WAV
                pcm_data = self._extract_pcm_from_wav(audio_data)
                if pcm_data:
                    audio_data = pcm_data
            
            # Convert to numpy array for processing
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Ensure mono (if stereo, convert to mono)
            if len(audio_array) % 2 == 0:  # Might be stereo
                # Try to reshape as stereo and convert to mono
                try:
                    stereo_array = audio_array.reshape(-1, 2)
                    audio_array = np.mean(stereo_array, axis=1).astype(np.int16)
                except:
                    # If reshape fails, assume it's already mono
                    pass
            
            # Resample if needed (Sarvam might return different sample rate)
            # For now, assume it's already 8kHz as requested
            
            # Convert back to bytes
            processed_audio = audio_array.tobytes()
            
            # Ensure frame alignment
            if len(processed_audio) % self.frame_size != 0:
                padding_needed = self.frame_size - (len(processed_audio) % self.frame_size)
                processed_audio += b'\x00' * padding_needed
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Error processing audio for playback: {str(e)}")
            return b''
    
    def _is_wav_format(self, audio_data: bytes) -> bool:
        """Check if audio data is in WAV format"""
        return audio_data.startswith(b'RIFF') and b'WAVE' in audio_data[:12]
    
    def _extract_pcm_from_wav(self, wav_data: bytes) -> Optional[bytes]:
        """Extract PCM data from WAV format"""
        try:
            # Use wave module to parse WAV
            with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
                # Get audio parameters
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                
                logger.info(f"WAV format: {sample_rate}Hz, {channels}ch, {sample_width*8}bit")
                
                # Read all frames
                pcm_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert to target format if needed
                if sample_rate != self.exotel_sample_rate or channels != 1 or sample_width != 2:
                    pcm_data = self._convert_audio_format(
                        pcm_data, sample_rate, channels, sample_width
                    )
                
                return pcm_data
                
        except Exception as e:
            logger.error(f"Error extracting PCM from WAV: {str(e)}")
            return None
    
    def _convert_audio_format(self, audio_data: bytes, src_rate: int, 
                             src_channels: int, src_width: int) -> bytes:
        """Convert audio to target format (8kHz, mono, 16-bit)"""
        try:
            # Convert to numpy array
            if src_width == 1:
                dtype = np.uint8
                audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
                audio_array = (audio_array - 128) / 128.0  # Convert to [-1, 1]
            elif src_width == 2:
                dtype = np.int16
                audio_array = np.frombuffer(audio_data, dtype=dtype).astype(np.float32)
                audio_array = audio_array / 32768.0  # Convert to [-1, 1]
            else:
                logger.error(f"Unsupported sample width: {src_width}")
                return audio_data
            
            # Handle stereo to mono conversion
            if src_channels == 2:
                audio_array = audio_array.reshape(-1, 2)
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample if needed (simple decimation/interpolation)
            if src_rate != self.exotel_sample_rate:
                # Simple resampling (not ideal but functional)
                ratio = self.exotel_sample_rate / src_rate
                new_length = int(len(audio_array) * ratio)
                
                if ratio < 1.0:  # Downsample
                    indices = np.linspace(0, len(audio_array) - 1, new_length).astype(int)
                    audio_array = audio_array[indices]
                else:  # Upsample
                    indices = np.linspace(0, len(audio_array) - 1, new_length)
                    audio_array = np.interp(indices, np.arange(len(audio_array)), audio_array)
            
            # Convert back to 16-bit integers
            audio_array = np.clip(audio_array * 32767, -32768, 32767).astype(np.int16)
            
            return audio_array.tobytes()
            
        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            return audio_data
    
    def validate_chunk_size(self, chunk_size: int) -> bool:
        """Validate if chunk size is appropriate for Exotel"""
        return (self.min_chunk_size <= chunk_size <= self.max_chunk_size and 
                chunk_size % self.frame_size == 0)
    
    def get_optimal_chunk_size(self, audio_length: int) -> int:
        """Get optimal chunk size for given audio length"""
        if audio_length <= self.min_chunk_size:
            return ((audio_length // self.frame_size) * self.frame_size)
        elif audio_length <= self.max_chunk_size:
            return audio_length
        else:
            return self.max_chunk_size
    
    def split_audio_into_chunks(self, audio_data: bytes, chunk_size: int = None) -> list:
        """Split audio data into appropriate chunks"""
        if not chunk_size:
            chunk_size = self.min_chunk_size
            
        chunks = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) % self.frame_size != 0:
                padding = self.frame_size - (len(chunk) % self.frame_size)
                chunk += b'\x00' * padding
            
            chunks.append(chunk)
        
        return chunks
    
    def create_wav_header(self, pcm_data_length: int) -> bytes:
        """Create WAV header for PCM data"""
        try:
            # WAV header structure
            chunk_size = 36 + pcm_data_length
            
            header = struct.pack('<4sI4s4sIHHIIHH4sI',
                b'RIFF',                    # ChunkID
                chunk_size,                 # ChunkSize
                b'WAVE',                    # Format
                b'fmt ',                    # Subchunk1ID
                16,                         # Subchunk1Size (PCM)
                1,                          # AudioFormat (PCM)
                self.exotel_channels,       # NumChannels
                self.exotel_sample_rate,    # SampleRate
                self.exotel_sample_rate * self.exotel_channels * self.exotel_sample_width,  # ByteRate
                self.exotel_channels * self.exotel_sample_width,  # BlockAlign
                self.exotel_sample_width * 8,  # BitsPerSample
                b'data',                    # Subchunk2ID
                pcm_data_length             # Subchunk2Size
            )
            
            return header
            
        except Exception as e:
            logger.error(f"Error creating WAV header: {str(e)}")
            return b''
    
    def encode_audio_base64(self, audio_data: bytes) -> str:
        """Encode audio data to base64 string"""
        try:
            return base64.b64encode(audio_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding audio to base64: {str(e)}")
            return ''
    
    def decode_audio_base64(self, encoded_audio: str) -> bytes:
        """Decode base64 audio string to bytes"""
        try:
            return base64.b64decode(encoded_audio)
        except Exception as e:
            logger.error(f"Error decoding audio from base64: {str(e)}")
            return b''
    
    def get_audio_duration(self, audio_data: bytes) -> float:
        """Calculate audio duration in seconds"""
        try:
            # For PCM: duration = samples / sample_rate
            # samples = bytes / (sample_width * channels)
            samples = len(audio_data) // (self.exotel_sample_width * self.exotel_channels)
            duration = samples / self.exotel_sample_rate
            return duration
        except Exception as e:
            logger.error(f"Error calculating audio duration: {str(e)}")
            return 0.0
    
    def normalize_audio(self, audio_data: bytes, target_level: float = 0.8) -> bytes:
        """Normalize audio to target level"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Calculate current peak level
            peak_level = np.max(np.abs(audio_array))
            
            if peak_level > 0:
                # Calculate scaling factor
                scale_factor = (target_level * 32767) / peak_level
                
                # Apply scaling and clip
                audio_array = np.clip(audio_array * scale_factor, -32768, 32767)
            
            # Convert back to int16
            return audio_array.astype(np.int16).tobytes()
            
        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return audio_data