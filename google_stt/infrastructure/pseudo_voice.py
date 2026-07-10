import threading
import time
from typing import Iterator
import numpy as np
import sounddevice as sd
from google_stt.domain.interfaces import TextToSpeech
from google_stt.domain.models import FillerSequence, PseudoVoicePitchMapper

class PseudoVoiceTTS(TextToSpeech):
    """サイン波による電子疑似音声および生成遅延時の非同期フィラー音を再生する具象実装"""
    def __init__(
        self,
        pitch_mapper: PseudoVoicePitchMapper,
        filler_sequence: FillerSequence,
        sample_rate: int = 44100,
        volume: float = 0.3,
        char_duration: float = 0.08,
        filler_duration: float = 0.16,
        filler_pause: float = 0.3
    ) -> None:
        self.pitch_mapper: PseudoVoicePitchMapper = pitch_mapper
        self.filler_sequence: FillerSequence = filler_sequence
        self.sample_rate: int = sample_rate
        self.volume: float = volume
        self.char_duration: float = char_duration
        self.filler_duration: float = filler_duration
        self.filler_pause: float = filler_pause
        self.is_speaking: bool = False

    def _make_beep(self, freq: float, duration: float) -> np.ndarray:
        n = int(self.sample_rate * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        env = np.ones(n)
        attack = int(n * 0.1)
        release = int(n * 0.3)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        return (np.sin(2 * np.pi * freq * t) * self.volume * env).astype(np.float32)

    def _play_filler_loop(self, stop_event: threading.Event) -> None:
        silence = np.zeros(int(self.sample_rate * self.filler_pause), dtype=np.float32)
        with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as out_stream:
            while not stop_event.is_set():
                c = self.filler_sequence.next_char()
                freq = self.pitch_mapper.map_char_to_frequency(c)
                if freq:
                    beep = self._make_beep(freq, duration=self.filler_duration)
                    out_stream.write(beep)
                else:
                    time.sleep(self.filler_duration)
                
                # シーケンスを1周ループしたらポーズを挟む
                if self.filler_sequence._current_index == 0:
                    out_stream.write(silence)

    def speak(self, text: str) -> None:
        self.is_speaking = True
        try:
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as out_stream:
                for c in text:
                    freq = self.pitch_mapper.map_char_to_frequency(c)
                    if freq:
                        beep = self._make_beep(freq, self.char_duration)
                        out_stream.write(beep)
                    else:
                        time.sleep(self.char_duration)
        finally:
            self.is_speaking = False

    def speak_stream(self, text_stream: Iterator[str]) -> None:
        self.is_speaking = True
        filler_stop = threading.Event()
        
        # フィラー音の非同期再生スレッドを開始
        self.filler_sequence.reset()
        filler_thread = threading.Thread(target=self._play_filler_loop, args=(filler_stop,), daemon=True)
        filler_thread.start()

        filler_stopped = False
        try:
            with sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype="float32") as out_stream:
                for char in text_stream:
                    # ストリーミングから最初のトークン（文字）が届いたらフィラーを停止
                    if not filler_stopped:
                        filler_stop.set()
                        filler_thread.join()  # スレッド停止を待つ（最大1音分の遅延）
                        filler_stopped = True

                    freq = self.pitch_mapper.map_char_to_frequency(char)
                    if freq:
                        beep = self._make_beep(freq, self.char_duration)
                        out_stream.write(beep)
                    else:
                        time.sleep(self.char_duration)
        finally:
            if not filler_stopped:
                filler_stop.set()
                filler_thread.join()
            self.is_speaking = False
