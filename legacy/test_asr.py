import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 5

print("モデル読み込み中...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("準備完了。5秒間録音します。")

input("Enterを押したら録音開始...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
print("認識中...")

segments, _ = model.transcribe(audio.flatten(), language="ja")
text = "".join(seg.text for seg in segments)
print(f"認識結果: {text}")
