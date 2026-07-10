from enum import Enum
from typing import List, Dict, Any, Optional

class Role(Enum):
    USER = "user"
    MODEL = "model"


class Message:
    """発話メッセージを表す値オブジェクト"""
    def __init__(self, role: Role, text: str) -> None:
        self.role: Role = role
        self.text: str = text

    def to_gemini_format(self) -> Dict[str, Any]:
        """Gemini API (Vertex AI) が要求するパーツ形式に変換する"""
        # 内部値は role.value が "user" または "model"
        # parts 内に text オブジェクトを持つ
        return {
            "role": self.role.value,
            "parts": [{"text": self.text}]
        }


class DialogueHistory:
    """会話の履歴と文脈を管理するエンティティ"""
    def __init__(self, system_instruction: Optional[str] = None) -> None:
        self.system_instruction: Optional[str] = system_instruction
        self._messages: List[Message] = []

    def add_user_message(self, text: str) -> None:
        """ユーザーの発話履歴を追加する。空文字は無視される。"""
        stripped_text = text.strip() if text else ""
        if stripped_text:
            self._messages.append(Message(Role.USER, stripped_text))

    def add_model_message(self, text: str) -> None:
        """モデルの発話履歴を追加する。空文字は無視される。"""
        stripped_text = text.strip() if text else ""
        if stripped_text:
            self._messages.append(Message(Role.MODEL, stripped_text))

    def get_messages(self) -> List[Message]:
        """現在のメッセージリストを取得する"""
        return self._messages

    def to_gemini_contents(self) -> List[Dict[str, Any]]:
        """Gemini APIのモデル入力用に履歴全体をフォーマット変換する"""
        return [msg.to_gemini_format() for msg in self._messages]

    def clear(self) -> None:
        """履歴を初期化する"""
        self._messages.clear()


class FillerSequence:
    """フィラー（言い淀み）の文字シーケンスおよびループ再生の状態を管理するエンティティ"""
    def __init__(self, sequence: List[str]) -> None:
        if not sequence:
            raise ValueError("Sequence cannot be empty")
        self.sequence: List[str] = sequence
        self._current_index: int = 0

    def next_char(self) -> str:
        """次のフィラー文字を取得し、内部インデックスを進める（終端に達したらループする）"""
        char = self.sequence[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.sequence)
        return char

    def reset(self) -> None:
        """再生状態を初期インデックスにリセットする"""
        self._current_index = 0


class PseudoVoicePitchMapper:
    """文字をサイン波の周波数に変換するドメインルールを司るドメインサービス"""
    def __init__(self, scale_frequencies: List[float], skip_characters: List[str]) -> None:
        self.scale_frequencies: List[float] = scale_frequencies
        self.skip_characters: set = set(skip_characters)

    def map_char_to_frequency(self, char: str) -> Optional[float]:
        """
        文字を対応する周波数（Hz）にマッピングする。
        スキップ対象文字（スペースなど）の場合は None を返す。
        """
        if not char or char in self.skip_characters:
            return None
        
        # 文字のUnicode値に基づき、指定の周波数リスト内でマッピング
        index = ord(char) % len(self.scale_frequencies)
        return self.scale_frequencies[index]
