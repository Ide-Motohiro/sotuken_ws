from typing import List, Iterator
from google_stt.domain.interfaces import (
    SpeechRecognizer, LanguageModel, TextToSpeech, FillerPlayer
)
from google_stt.domain.models import DialogueHistory

class MockSpeechRecognizer(SpeechRecognizer):
    """音声認識 (ASR) のテスト用モック"""
    def __init__(self, texts: List[str] = None) -> None:
        self.texts: List[str] = texts if texts is not None else []
        self._current_index: int = 0

    def recognize_once(self) -> str:
        if self._current_index < len(self.texts):
            text = self.texts[self._current_index]
            self._current_index += 1
            return text
        return ""


class MockLanguageModel(LanguageModel):
    """対話モデル (LLM) のテスト用モック"""
    def __init__(self, reply: str = "Mock Reply") -> None:
        self.reply: str = reply

    def generate_reply(self, history: DialogueHistory) -> str:
        return self.reply

    def generate_reply_stream(self, history: DialogueHistory) -> Iterator[str]:
        # 返答テキストを1文字ずつ返すジェネレータ
        for char in self.reply:
            yield char


class MockTextToSpeech(TextToSpeech):
    """音声合成・出力 (TTS) のテスト用モック"""
    def __init__(self) -> None:
        self.spoken_texts: List[str] = []

    def speak(self, text: str) -> None:
        self.spoken_texts.append(text)

    def speak_stream(self, text_stream: Iterator[str]) -> None:
        # ストリームのテキストを全て結合して記録する
        full_text = "".join(text_stream)
        self.spoken_texts.append(full_text)


class MockFillerPlayer(FillerPlayer):
    """フィラー再生のテスト用モック。呼び出し順を記録する。

    events は外から差し替えられる（TTSの合成・再生と同じ列に混ぜて順序を検証するため）。
    冪等性は本実装と同じく内部状態で判定する。イベント列から推測すると、間に別の
    イベントが挟まったときに二重記録される。
    """
    def __init__(self) -> None:
        self.events: List[str] = []
        self._running: bool = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.events.append("start")

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.events.append("stop")
