"""
Voice Activity Detection (VAD) Engine
Provides both energy-based and WebRTC-based VAD for detecting speech in audio streams.
Supports barge-in (interruption) detection.
"""

import struct
import numpy as np
from typing import Tuple, Optional

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


class VADEngine:
    """Voice Activity Detection engine with barge-in support."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        energy_threshold: float = 0.01,
        silence_duration_ms: int = 1200,
        speech_min_duration_ms: int = 250,
    ):
        """
        Initialize VAD engine.

        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame size in ms (10, 20, or 30)
            aggressiveness: WebRTC VAD aggressiveness (0-3, higher = more aggressive)
            energy_threshold: RMS energy threshold for energy-based VAD
            silence_duration_ms: Duration of silence to consider speech ended
            speech_min_duration_ms: Minimum speech duration to trigger
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.silence_duration_ms = silence_duration_ms
        self.speech_min_duration_ms = speech_min_duration_ms

        # Frame size in samples
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # WebRTC VAD
        self.vad = None
        if WEBRTC_AVAILABLE:
            self.vad = webrtcvad.Vad(aggressiveness)

        # State tracking
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.frames_for_silence = int(silence_duration_ms / frame_duration_ms)
        self.frames_for_speech = int(speech_min_duration_ms / frame_duration_ms)

        # Buffer for accumulating audio
        self._buffer = bytearray()

    def reset(self):
        """Reset VAD state."""
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self._buffer = bytearray()

    def process_audio_chunk(self, audio_data: bytes) -> dict:
        """
        Process an audio chunk and return VAD result.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono)

        Returns:
            dict with:
                - is_speech: bool - whether current frame contains speech
                - speech_started: bool - whether speech just started
                - speech_ended: bool - whether speech just ended (silence detected)
                - energy: float - RMS energy level (0.0-1.0)
                - is_speaking: bool - overall speaking state
        """
        self._buffer.extend(audio_data)

        result = {
            "is_speech": False,
            "speech_started": False,
            "speech_ended": False,
            "energy": 0.0,
            "is_speaking": self.is_speaking,
        }

        # Process complete frames
        frame_bytes = self.frame_size * 2  # 16-bit = 2 bytes per sample

        while len(self._buffer) >= frame_bytes:
            frame = bytes(self._buffer[:frame_bytes])
            self._buffer = self._buffer[frame_bytes:]

            # Calculate energy
            energy = self._calculate_energy(frame)
            result["energy"] = energy

            # Detect speech
            is_speech = self._detect_speech(frame, energy)
            result["is_speech"] = is_speech

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0

                if not self.is_speaking and self.speech_frames >= self.frames_for_speech:
                    self.is_speaking = True
                    result["speech_started"] = True
            else:
                self.silence_frames += 1

                if self.is_speaking and self.silence_frames >= self.frames_for_silence:
                    self.is_speaking = False
                    self.speech_frames = 0
                    result["speech_ended"] = True

            result["is_speaking"] = self.is_speaking

        return result

    def detect_barge_in(self, audio_data: bytes) -> bool:
        """
        Quick check if audio contains speech (for barge-in/interruption detection).
        Uses a lower threshold for faster response.

        Args:
            audio_data: Raw PCM audio bytes

        Returns:
            True if speech/voice activity detected
        """
        energy = self._calculate_energy(audio_data)
        # Lower threshold for barge-in to be more responsive
        barge_in_threshold = self.energy_threshold * 0.7

        if energy > barge_in_threshold:
            if self.vad and len(audio_data) == self.frame_size * 2:
                try:
                    return self.vad.is_speech(audio_data, self.sample_rate)
                except Exception:
                    return True
            return True
        return False

    def _detect_speech(self, frame: bytes, energy: float) -> bool:
        """Detect if a frame contains speech using both energy and WebRTC VAD."""
        # Energy-based detection
        energy_speech = energy > self.energy_threshold

        # WebRTC VAD detection
        if self.vad:
            try:
                webrtc_speech = self.vad.is_speech(frame, self.sample_rate)
                # Combine both: either one detecting speech counts
                score = 0

                if energy_speech:
                    score += 0.4

                if webrtc_speech:
                    score += 0.7

                return score >= 0.7
            except Exception:
                return energy_speech
        return energy_speech

    def _calculate_energy(self, frame: bytes) -> float:
        """Calculate RMS energy of an audio frame."""
        if len(frame) < 2:
            return 0.0

        # Convert bytes to int16 samples
        n_samples = len(frame) // 2
        try:
            samples = struct.unpack(f"<{n_samples}h", frame[:n_samples * 2])
        except struct.error:
            return 0.0

        # Calculate RMS
        arr = np.array(samples, dtype=np.float64)
        rms = np.sqrt(np.mean(arr ** 2)) / 32768.0

        return float(rms)

    def get_state(self) -> dict:
        """Get current VAD state."""
        return {
            "is_speaking": self.is_speaking,
            "speech_frames": self.speech_frames,
            "silence_frames": self.silence_frames,
            "webrtc_available": WEBRTC_AVAILABLE,
        }


class BargeinDetector:
    """
    Specialized detector for barge-in (user interruption) during TTS playback.
    More sensitive and faster response than regular VAD.
    """

    def __init__(self, sample_rate: int = 16000, sensitivity: float = 0.6):
        """
        Args:
            sample_rate: Audio sample rate
            sensitivity: 0.0-1.0, higher = more sensitive to interruption
        """
        self.sample_rate = sample_rate
        self.sensitivity = sensitivity
        self.energy_threshold = 0.008 * (1.1 - sensitivity)
        self.consecutive_speech_frames = 0
        self.required_frames = max(2, int(5 * (1 - sensitivity)))

    def reset(self):
        """Reset detector state."""
        self.consecutive_speech_frames = 0

    def check(self, audio_data: bytes) -> bool:
        """
        Check if audio contains a barge-in attempt.

        Args:
            audio_data: Raw PCM audio bytes (16-bit mono)

        Returns:
            True if user is trying to interrupt
        """
        if len(audio_data) < 2:
            return False

        n_samples = len(audio_data) // 2
        try:
            samples = struct.unpack(f"<{n_samples}h", audio_data[:n_samples * 2])
        except struct.error:
            return False

        arr = np.array(samples, dtype=np.float64)
        rms = np.sqrt(np.mean(arr ** 2)) / 32768.0

        if rms > self.energy_threshold:
            self.consecutive_speech_frames += 1
        else:
            self.consecutive_speech_frames = 0

        return self.consecutive_speech_frames >= self.required_frames
