import queue
import threading
import time

from dotenv import find_dotenv, load_dotenv
from google.cloud import speech

load_dotenv(find_dotenv())

SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)
STABILITY_DURATION = 0.4  # 暫定結果がこの秒数変化しなければ確定扱い

_stt_client = speech.SpeechClient()
_streaming_config = speech.StreamingRecognitionConfig(
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE,
        language_code="ja-JP",
    ),
    interim_results=True,
)

audio_queue: queue.Queue = queue.Queue()


def make_callback(is_speaking_fn):
    """is_speaking_fn: () -> bool。Trueのときマイク入力をスキップする"""
    def callback(indata, _frames, _timestamp, _status):
        if not is_speaking_fn():
            audio_queue.put(bytes(indata))
    return callback


def recognize_once() -> str:
    stop_streaming = threading.Event()
    last_interim = ""
    stable_since = 0.0

    def generate_requests():
        while not stop_streaming.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    responses = _stt_client.streaming_recognize(_streaming_config, generate_requests())
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
            elif transcript and (now - stable_since) >= STABILITY_DURATION:
                stop_streaming.set()
                return transcript
            print(f"\r途中: {transcript}", end="", flush=True)
    return ""
