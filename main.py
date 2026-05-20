import io
import os
import queue
import tempfile
import wave

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import requests
import sounddevice as sd
import winsound
from groq import Groq

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3

SILENCE_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
MIN_CHUNKS = int(MIN_SPEECH_DURATION / CHUNK_DURATION)

VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん ノーマル

SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"

client = Groq(api_key=os.environ["GROQ_API_KEY"])
conversation_history = []

audio_queue = queue.Queue()
audio_buffer = []
is_recording = False
silence_count = 0
is_speaking = False


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def transcribe(audio: np.ndarray) -> str:
    result = client.audio.transcriptions.create(
        file=("audio.wav", to_wav_bytes(audio)),
        model="whisper-large-v3-turbo",
        language="ja",
    )
    return result.text.strip()


def chat(user_text: str) -> str:
    conversation_history.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply


def speak(text: str) -> None:
    global is_speaking
    is_speaking = True
    try:
        query = requests.post(
            f"{VOICEVOX_URL}/audio_query",
            params={"text": text, "speaker": SPEAKER_ID},
        ).json()
        audio = requests.post(
            f"{VOICEVOX_URL}/synthesis",
            params={"speaker": SPEAKER_ID},
            json=query,
        ).content
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            tmp = f.name
        winsound.PlaySound(tmp, winsound.SND_FILENAME)
        os.unlink(tmp)
    finally:
        is_speaking = False


def callback(indata, frames, time, status):
    global is_recording, silence_count, audio_buffer

    if is_speaking:
        return

    chunk = indata[:, 0].copy()
    rms = float(np.sqrt(np.mean(chunk ** 2)))

    if rms > SILENCE_THRESHOLD:
        if not is_recording:
            print("🎤 発話検出")
            is_recording = True
        audio_buffer.append(chunk)
        silence_count = 0
    elif is_recording:
        audio_buffer.append(chunk)
        silence_count += 1
        if silence_count >= SILENCE_CHUNKS:
            if len(audio_buffer) >= MIN_CHUNKS:
                audio_queue.put(np.concatenate(audio_buffer))
            audio_buffer.clear()
            is_recording = False
            silence_count = 0


print("準備完了。話しかけてください。（Ctrl+C で終了）\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=CHUNK_SIZE, callback=callback):
    while True:
        audio = audio_queue.get()

        print("認識中...")
        user_text = transcribe(audio)
        if not user_text:
            continue
        print(f"あなた: {user_text}")

        print("考え中...")
        reply = chat(user_text)
        print(f"AI: {reply}\n")

        speak(reply)
