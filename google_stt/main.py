import os
import queue
import tempfile
import winsound

import requests
import sounddevice as sd
from dotenv import find_dotenv, load_dotenv
from google import genai
from google.cloud import speech

load_dotenv(find_dotenv())

SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)
VOICEVOX_URL = "http://localhost:50021"
SPEAKER_ID = 3  # ずんだもん ノーマル
SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"

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


def callback(indata, frames, time, status):
    if not is_speaking:
        audio_queue.put(bytes(indata))


def recognize_once() -> str:
    def generate_requests():
        while True:
            chunk = audio_queue.get()
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    responses = stt_client.streaming_recognize(streaming_config, generate_requests())
    for response in responses:
        for result in response.results:
            transcript = result.alternatives[0].transcript
            if result.is_final:
                return transcript
            print(f"\r途中: {transcript}", end="", flush=True)
    return ""


def chat(user_text: str) -> str:
    conversation_history.append({"role": "user", "parts": [{"text": user_text}]})
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=conversation_history,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    reply = response.text.strip()
    conversation_history.append({"role": "model", "parts": [{"text": reply}]})
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


print("準備完了。話しかけてください。（Ctrl+C で終了）\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=CHUNK_SIZE, callback=callback):
    while True:
        user_text = recognize_once().strip()
        if not user_text:
            continue
        print(f"\rあなた: {user_text}")

        print("考え中...")
        reply = chat(user_text)
        print(f"AI: {reply}\n")

        speak(reply)
