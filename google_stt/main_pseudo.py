import queue
import threading
import time

import numpy as np
import sounddevice as sd
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.cloud import speech

load_dotenv(find_dotenv())

SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)
SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"

PSEUDO_SAMPLE_RATE = 44100
PSEUDO_VOLUME = 0.3
CHAR_DURATION = 0.08  # 1文字あたりの秒数

# ペンタトニックスケール（どの文字の組み合わせでも不協和にならない）
SCALE_FREQS = [261.6, 329.6, 392.0, 523.3, 659.3, 784.0, 1046.5]

SKIP_CHARS = set(" 　\n\r\t")

stt_client = speech.SpeechClient()
llm_client = genai.Client(
    vertexai=True,
    project="gen-lang-client-0698198570",
    location="us-central1",
)

stt_config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=SAMPLE_RATE,
    language_code="ja-JP",
)
streaming_config = speech.StreamingRecognitionConfig(
    config=stt_config,
    interim_results=True,
)

conversation_history = []
is_speaking = False
audio_queue = queue.Queue()


def callback(indata, _frames, _timestamp, _status):
    if not is_speaking:
        audio_queue.put(bytes(indata))


STABILITY_DURATION = 0.4  # 暫定結果がこの秒数変化しなければ確定扱い


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

    responses = stt_client.streaming_recognize(streaming_config, generate_requests())
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


def char_to_freq(c: str) -> float:
    return SCALE_FREQS[ord(c) % len(SCALE_FREQS)]


def make_beep(freq: float) -> np.ndarray:
    n = int(PSEUDO_SAMPLE_RATE * CHAR_DURATION)
    t = np.linspace(0, CHAR_DURATION, n, endpoint=False)
    env = np.ones(n)
    attack = int(n * 0.1)
    release = int(n * 0.3)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return (np.sin(2 * np.pi * freq * t) * PSEUDO_VOLUME * env).astype(np.float32)


def chat_with_pseudo_voice(user_text: str) -> str:
    global is_speaking

    char_queue: queue.Queue[str] = queue.Queue()
    done_event = threading.Event()

    def run_pseudo_voice():
        with sd.OutputStream(samplerate=PSEUDO_SAMPLE_RATE, channels=1, dtype="float32") as out_stream:
            while True:
                try:
                    c = char_queue.get(timeout=0.05)
                except queue.Empty:
                    if done_event.is_set():
                        break
                    continue
                print(c, end="", flush=True)
                if c not in SKIP_CHARS:
                    out_stream.write(make_beep(char_to_freq(c)))

    conversation_history.append({"role": "user", "parts": [{"text": user_text}]})
    stream = llm_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    pseudo_thread = threading.Thread(target=run_pseudo_voice, daemon=True)
    full_text = ""
    print("AI: ", end="", flush=True)

    for chunk in stream:
        if not chunk.text:
            continue

        if not pseudo_thread.is_alive():
            is_speaking = True
            pseudo_thread.start()

        for c in chunk.text:
            char_queue.put(c)

        full_text += chunk.text

    done_event.set()
    pseudo_thread.join()
    is_speaking = False
    print()

    conversation_history.append({"role": "model", "parts": [{"text": full_text}]})
    return full_text


print("準備完了。話しかけてください。（Ctrl+C で終了）\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=CHUNK_SIZE, callback=callback):
    while True:
        user_text = recognize_once().strip()
        if not user_text:
            continue
        print(f"\rあなた: {user_text}")
        chat_with_pseudo_voice(user_text)
