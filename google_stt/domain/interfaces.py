from abc import ABC, abstractmethod
from typing import Iterator, Optional
from google_stt.domain.models import DialogueHistory

class SpeechRecognizer(ABC):
    """音声認識 (ASR) を抽象化するインターフェース"""

    #: 直近の発話で、終端と判定するまでに待った秒数。測らない実装では None のまま。
    #: 体感される無音はユーザーが喋り終わった瞬間から始まるので、この区間も内訳に要る。
    last_endpoint_wait_sec: Optional[float] = None

    #: 直近の認識結果に付いていた信頼度。取れない実装・経路では None のまま。
    last_confidence: Optional[float] = None
    
    @abstractmethod
    def recognize_once(self) -> str:
        """
        マイク入力などから音声を取得し、1回分の発話が確定するまでブロックし、
        認識されたテキストを返す。
        """
        pass


class LanguageModel(ABC):
    """対話生成モデル (LLM) を抽象化するインターフェース"""

    #: 直近の生成にかかった秒数。ストリーミングでは最初の断片が届くまで。
    #: 測らない実装では None のまま。
    last_generation_sec: Optional[float] = None

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

    #: 直近の発話で、speak() が呼ばれてから最初の音が鳴るまでの秒数。
    #: **フィラーで隠すべき区間のうち TTS が占める分**。測らない実装では None のまま。
    last_time_to_first_sound_sec: Optional[float] = None

    #: 直近の発話の再生時間（秒）。測らない実装では None のまま。
    last_playback_sec: Optional[float] = None

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


class FillerPlayer(ABC):
    """応答生成の待機中に鳴らすフィラー（言い淀み）音を抽象化するインターフェース。

    フィラーの有無は本研究の独立変数そのものなので、条件間で挙動がぶれないよう
    静的な実装に留めること（生成内容やタイミングを動的に変えない）。

    埋めるべき区間は「ユーザーの発話終端を検出してから、応答の音が実際に鳴り始めるまで」。
    実測ではこれが約2.4秒で、内訳は応答生成 約0.9秒 ＋ 音声合成 約1.1秒（残りは終端判定）。
    音声合成の分を含めるのが要点で、応答テキストが返った時点で止めると約1.1秒の穴が空く。
    """

    @abstractmethod
    def start(self) -> None:
        """フィラーの再生を開始する。呼び出しはブロックしない。"""
        pass

    @abstractmethod
    def stop(self) -> None:
        """フィラーの再生を停止する。冪等であること（複数回呼ばれても安全）。"""
        pass
