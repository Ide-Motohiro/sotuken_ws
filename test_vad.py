import io
import os
import queue
import wave

import numpy as np
import sounddevice as sd
from groq import Groq

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3

SILENCE_CHUNKS = int(SILENCE_DURATION / CHUNK_DURATION)
MIN_CHUNKS = int(MIN_SPEECH_DURATION / CHUNK_DURATION)

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

audio_queue = queue.Queue()
audio_buffer = []
is_recording = False
silence_count = 0

def callback(indata, frames, time, status):
    global is_recording, silence_count, audio_buffer

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

print("準備完了。話しかけてください。\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=CHUNK_SIZE, callback=callback):
    while True:
        audio = audio_queue.get()
        print("認識中...")
        result = client.audio.transcriptions.create(
            file=("audio.wav", to_wav_bytes(audio)),
            model="whisper-large-v3-turbo",
            language="ja",
        )
        text = result.text.strip()
        if text:
            print(f"認識結果: {text}\n")
