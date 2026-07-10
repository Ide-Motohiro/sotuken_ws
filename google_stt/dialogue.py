import queue
import threading

import numpy as np
import sounddevice as sd
from dotenv import find_dotenv, load_dotenv
from google import genai

load_dotenv(find_dotenv())

PSEUDO_SAMPLE_RATE = 44100
PSEUDO_VOLUME = 0.3
CHAR_DURATION = 0.08
SCALE_FREQS = [261.6, 329.6, 392.0, 523.3, 659.3, 784.0, 1046.5]
SKIP_CHARS = set(" 　\n\r\t")
SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"

FILLER_DURATION = CHAR_DURATION * 2       # フィラーの1音の長さ（通常より遅め）
FILLER_PAUSE = 0.3                         # フィラー音列の後の無音
FILLER_SEQUENCE = ["え", "ー", "と"]      # ループするフィラー音列

_llm_client = genai.Client(
    vertexai=True,
    project="gen-lang-client-0698198570",
    location="us-central1",
)

conversation_history: list = []
is_speaking: bool = False


def char_to_freq(c: str) -> float:
    return SCALE_FREQS[ord(c) % len(SCALE_FREQS)]


def make_beep(freq: float, duration: float = CHAR_DURATION) -> np.ndarray:
    n = int(PSEUDO_SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    env = np.ones(n)
    attack = int(n * 0.1)
    release = int(n * 0.3)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return (np.sin(2 * np.pi * freq * t) * PSEUDO_VOLUME * env).astype(np.float32)


def _play_filler(stop_event: threading.Event) -> None:
    silence = np.zeros(int(PSEUDO_SAMPLE_RATE * FILLER_PAUSE), dtype=np.float32)
    with sd.OutputStream(samplerate=PSEUDO_SAMPLE_RATE, channels=1, dtype="float32") as out_stream:
        i = 0
        while not stop_event.is_set():
            c = FILLER_SEQUENCE[i % len(FILLER_SEQUENCE)]
            out_stream.write(make_beep(char_to_freq(c), duration=FILLER_DURATION))
            i += 1
            if i % len(FILLER_SEQUENCE) == 0:
                out_stream.write(silence)


def chat_with_pseudo_voice(user_text: str) -> str:
    global is_speaking

    # フィラーを即時開始（Geminiのレイテンシを隠す）
    is_speaking = True
    filler_stop = threading.Event()
    filler_thread = threading.Thread(target=_play_filler, args=(filler_stop,), daemon=True)
    filler_thread.start()

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
    stream = _llm_client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config={"system_instruction": SYSTEM_PROMPT},
    )

    pseudo_thread = threading.Thread(target=run_pseudo_voice, daemon=True)
    full_text = ""
    filler_stopped = False
    print("AI: ", end="", flush=True)

    for chunk in stream:
        if not chunk.text:
            continue

        if not filler_stopped:
            filler_stop.set()
            filler_thread.join()  # 最大1音分（約160ms）待つ
            filler_stopped = True

        if not pseudo_thread.is_alive():
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
