from abc import ABC, abstractmethod
from typing import Iterator
from google_stt.domain.models import DialogueHistory

class SpeechRecognizer(ABC):
    """音声認識 (ASR) を抽象化するインターフェース"""
    
    @abstractmethod
    def recognize_once(self) -> str:
        """
        マイク入力などから音声を取得し、1回分の発話が確定するまでブロックし、
        認識されたテキストを返す。
        """
        pass


class LanguageModel(ABC):
    """対話生成モデル (LLM) を抽象化するインターフェース"""

    @abstractmethod
    def generate_reply(self, history: DialogueHistory) -> str:
        """会話履歴から一括で返答テキストを生成する"""
        pass

    @abstractmethod
    def generate_reply_stream(self, history: DialogueHistory) -> Iterator[str]:
        """会話履歴から、文字またはトークン単位で逐次的に返答テキストをストリーミング生成する"""
        pass


class TextToSpeech(ABC):
    """音声合成および音声出力 (TTS) を抽象化するインターフェース"""

    @abstractmethod
    def speak(self, text: str) -> None:
        """テキストを音声に変換し、再生完了するまでブロックして出力する"""
        pass

    @abstractmethod
    def speak_stream(self, text_stream: Iterator[str]) -> None:
        """
        テキストストリームをリアルタイムに音声へ変換し出力する。
        （疑似音声のサイン波再生や、TTSストリーミング向け）
        """
        pass
