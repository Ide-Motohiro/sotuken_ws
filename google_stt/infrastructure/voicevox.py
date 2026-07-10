import os
import tempfile
from typing import Iterator
import requests
import winsound
from google_stt.domain.interfaces import TextToSpeech

class VoiceVoxTTS(TextToSpeech):
    """VOICEVOX サーバーを用いた音声合成・再生の具象実装"""
    def __init__(self, url: str = "http://localhost:50021", speaker_id: int = 3) -> None:
        self.url: str = url
        self.speaker_id: int = speaker_id
        self.is_speaking: bool = False

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self.is_speaking = True
        try:
            query = requests.post(
                f"{self.url}/audio_query",
                params={"text": text, "speaker": self.speaker_id},
            ).json()
            audio = requests.post(
                f"{self.url}/synthesis",
                params={"speaker": self.speaker_id},
                json=query,
            ).content
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                tmp = f.name
            winsound.PlaySound(tmp, winsound.SND_FILENAME)
            os.unlink(tmp)
        finally:
            self.is_speaking = False

    def speak_stream(self, text_stream: Iterator[str]) -> None:
        """ストリームテキストを一括結合してVOICEVOXで再生する"""
        full_text = "".join(text_stream)
        self.speak(full_text)
