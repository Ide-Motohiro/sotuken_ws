import requests
import winsound
import tempfile
import os

VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん ノーマル

text = "こんにちは、ずんだもんなのだ。テスト成功なのだ。"

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
