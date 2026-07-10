import queue
import threading
import time
import sounddevice as sd
from google.cloud import speech
from google_stt.domain.interfaces import SpeechRecognizer

class GoogleSpeechRecognizer(SpeechRecognizer):
    """Google Cloud Speech-to-Text (Streaming API) を使った音声認識具象実装"""
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,
        stability_duration: float = 0.4,
        is_speaking_fn = lambda: False
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.stability_duration: float = stability_duration
        self.is_speaking_fn = is_speaking_fn
        self.audio_queue: queue.Queue = queue.Queue()
        
        self._stt_client = speech.SpeechClient()
        self._streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="ja-JP",
            ),
            interim_results=True,
        )

    def _callback(self, indata, _frames, _timestamp, _status):
        if not self.is_speaking_fn():
            self.audio_queue.put(bytes(indata))

    def recognize_once(self) -> str:
        """マイク入力を開始し、発話が確定（stability_durationの間変化なし）するまでブロックして文字列を返す"""
        # 前の発話で残った古いキューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        stop_streaming = threading.Event()
        last_interim = ""
        stable_since = 0.0

        def generate_requests():
            while not stop_streaming.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        # マイク入力を開始して Google STT にストリーミング送信
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                            blocksize=self.chunk_size, callback=self._callback):
            responses = self._stt_client.streaming_recognize(self._streaming_config, generate_requests())
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        stop_streaming.set()
                        return transcript
                    
                    now = time.monotonic()
                    if transcript != last_interim:
                        last_interim = transcript
                        stable_since = now
                    elif transcript and (now - stable_since) >= self.stability_duration:
                        stop_streaming.set()
                        return transcript
                    print(f"\r途中: {transcript}", end="", flush=True)
        return ""
