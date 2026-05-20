import os
import tempfile

import requests
import winsound
from dotenv import find_dotenv, load_dotenv
from groq import Groq
from RealtimeSTT import AudioToTextRecorder

load_dotenv(find_dotenv())

VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん ノーマル

SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"

client = Groq(api_key=os.environ["GROQ_API_KEY"])
conversation_history = []


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
    global recorder
    recorder.set_microphone(False)
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
        recorder.set_microphone(True)


def on_text(text: str) -> None:
    text = text.strip()
    if not text:
        return
    print(f"あなた: {text}")
    print("考え中...")
    reply = chat(text)
    print(f"AI: {reply}\n")
    speak(reply)


if __name__ == "__main__":
    print("準備完了。話しかけてください。（Ctrl+C で終了）\n")
    recorder = AudioToTextRecorder(language="ja", model="small")
    while True:
        recorder.text(on_text)
