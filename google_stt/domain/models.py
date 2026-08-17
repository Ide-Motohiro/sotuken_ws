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


#: 日本語の子音を調音様式・有声性でまとめたグループ。この中で置換すると調音点だけが変わる。
#: 拗音（ky, gy, ny …）を平音と混ぜるとモーラの構造自体が変わるので別グループにしてある。
#: 音素の表記は VOICEVOX の audio_query が用いるローマ字表記に合わせている
#: （収録している30種は VOICEVOX で合成が通ることを実測で確認済み。"ng" と "q" は通らない）。
ARTICULATION_GROUPS: List[List[str]] = [
    ["k", "t", "p"],      # 無声破裂音
    ["ky", "ty", "py"],   # 無声破裂音（拗音）
    ["g", "d", "b"],      # 有声破裂音
    ["gy", "dy", "by"],   # 有声破裂音（拗音）
    ["s", "h", "f"],      # 無声摩擦音
    ["sh", "hy"],         # 無声摩擦音（硬口蓋寄り）
    ["z", "j", "v"],      # 有声摩擦音・破擦音
    ["ch", "ts"],         # 無声破擦音
    ["n", "m"],           # 鼻音
    ["ny", "my"],         # 鼻音（拗音）
    ["r", "w"],           # 流音・接近音
    ["y", "ry"],          # 硬口蓋接近音系
]

#: 置換の対象になる子音の全体。ARTICULATION_GROUPS を平坦化したもの。
CONSONANTS: List[str] = [c for group in ARTICULATION_GROUPS for c in group]


class ConsonantSubstitutionTable:
    """子音を別の子音へ写す固定置換表と、どのモーラを差し替えるかの規則を持つ値オブジェクト。

    疑似言語音（意味は取れないが発話のリズム・抑揚は残る音声）を作るためのドメインルール。
    音素の識別だけを壊し、韻律・リズムには手を触れないという方針をここで表現する。

    **置換表は固定である**ことが本質的で、同じ子音は常に同じ子音へ写る。発話のたびに
    ランダムに選び直すと規則性が失われ、言語ではなく喃語として知覚される（聴取比較で確認）。

    `swap_ratio` は差し替えるモーラの割合で、了解性のつまみになる。1.0 で全モーラ、
    0.5 で1モーラおき。子音を持つモーラだけを通し番号で数え、位置だけで決まる規則で選ぶため、
    入力テキストの内容では分岐せず、同じ文なら常に同じ結果になる。
    """

    def __init__(self, mapping: Dict[str, str], swap_ratio: float = 0.5) -> None:
        if not 0.0 <= swap_ratio <= 1.0:
            raise ValueError(f"swap_ratio は 0.0〜1.0 で指定してください: {swap_ratio}")
        fixed_points = sorted(src for src, dst in mapping.items() if src == dst)
        if fixed_points:
            # 自分自身へ写る子音があると、そこだけ加工が効かず素のまま鳴ってしまう
            raise ValueError(f"置換表に自分自身へ写る子音があります: {fixed_points}")
        self.mapping: Dict[str, str] = dict(mapping)
        self.swap_ratio: float = swap_ratio

    @classmethod
    def articulatory(cls, swap_ratio: float = 0.5) -> "ConsonantSubstitutionTable":
        """調音様式・有声性を保ったまま調音点だけを変える置換表（k→t、n→m など）。

        「別の言語」ではなく「滑舌が悪い」側に寄るため、意味を壊しつつ聞き疲れしにくい。
        聴取比較では swap_ratio=0.5 との組み合わせが最も手応えがあった。
        """
        mapping = {}
        for group in ARTICULATION_GROUPS:
            for i, consonant in enumerate(group):
                mapping[consonant] = group[(i + 1) % len(group)]
        return cls(mapping, swap_ratio)

    @classmethod
    def distant(cls, swap_ratio: float = 1.0) -> "ConsonantSubstitutionTable":
        """調音的な近さを無視し、全子音の環を半周ずらす置換表（比較対照用）。

        子音数が偶数（30）なので、どの子音も自分自身には写らない。
        意味は確実に消えるが、聴取比較では「別の言語」に聞こえ了解性が大きく落ちた。
        """
        n = len(CONSONANTS)
        mapping = {c: CONSONANTS[(i + n // 2) % n] for i, c in enumerate(CONSONANTS)}
        return cls(mapping, swap_ratio)

    def substitute(self, consonant: str) -> Optional[str]:
        """子音の置換先を返す。表に無い子音は None を返す（呼び出し側で素通しさせる）"""
        return self.mapping.get(consonant)

    def should_swap(self, index: int) -> bool:
        """子音を持つモーラの通し番号 index を差し替えるかどうかを返す。

        swap_ratio の割合が全体に均等に散るよう、累積個数が繰り上がる位置だけを選ぶ。
        swap_ratio=0.5 なら 0, 2, 4, ... 番目が対象になる。
        """
        if self.swap_ratio <= 0.0:
            return False
        if index < 0:
            raise ValueError(f"index は0以上で指定してください: {index}")
        if index == 0:
            return True
        return int(index * self.swap_ratio) > int((index - 1) * self.swap_ratio)


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
