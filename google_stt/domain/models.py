from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional

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




@dataclass(frozen=True)
class TurnTiming:
    """1ターンの時間の内訳（秒）。ユーザーが喋り終わってから応答の音が出るまでを分解する。

    体感リアルタイム性が本研究の従属変数そのものなので、**本番の経路の中で測れること**が要る。
    区間ごとの測定はそれぞれの実装（認識器・モデル・TTS）が持ち、ここではそれを束ねるだけ。
    測っていない区間は None のまま入る。

    **`endpoint_wait_sec` の起点は「中間認識結果が変化しなくなった時刻」であり、
    ユーザーが実際に口を閉じた時刻ではない。** 認識サービスの中間結果は音声から遅れて届くため、
    その遅れ（未計測）の分だけ実際の無音はここに現れる値より長い。論文で述べるときは
    この留保を落とさないこと。
    """
    endpoint_wait_sec: Optional[float] = None
    generation_sec: Optional[float] = None
    time_to_first_sound_sec: Optional[float] = None
    playback_sec: Optional[float] = None
    #: フィラーを鳴らし終わるまで応答を待たせた秒数（フィラー無し条件では None）
    filler_stop_delay_sec: Optional[float] = None

    @property
    def time_to_response_sec(self) -> Optional[float]:
        """発話終端の検出から応答の音が鳴り始めるまで。欠けている区間があれば None。

        フィラーあり条件ではこの区間の一部にフィラーが鳴るので、全部が無音とは限らない。
        """
        parts = (self.endpoint_wait_sec, self.generation_sec, self.time_to_first_sound_sec)
        if any(part is None for part in parts):
            return None
        return sum(parts) + (self.filler_stop_delay_sec or 0.0)


#: これで終わっている発話は、まだ続きがあると見なす語尾。
#:
#: 日本語の発話中の休止は文節末（助詞・接続助詞の直後）に集中する。
#: 「えーと、昨日ね……映画を見たんだけど」の「ね」のあとで切られると発話が途中で確定してしまう。
#: 意味を解釈するのではなく、**語の終わり方だけ**を見る単純な規則にしてある
#: （解釈を挟むと条件間で挙動がぶれるうえ、認識結果は句読点を持たないため当てにならない）。
#:
#: 長い方の待ちに倒れても損は待ち時間だけなので、迷ったら入れる方針で選んである。
CONTINUATION_ENDINGS: Tuple[str, ...] = (
    # 接続助詞（前件と後件をつなぐので、後件がまだ来ていない）
    "て", "で", "けど", "けれど", "けれども", "が", "し", "から", "ので", "のに",
    "たら", "なら", "ば", "とか", "って", "という",
    # 格助詞・係助詞（体言に付くので、述語がまだ来ていない）
    "は", "も", "を", "に", "へ", "と", "の", "や",
    # 言いよどみ（次の語を探している最中）
    "えーと", "えっと", "ええと", "あの", "あのー", "その", "まあ", "なんか",
    # 述語を要求する副詞（後ろに来る述語がまだ来ていない）
    # 「本読むのってやっぱ……おもしろいよな」の「やっぱ」で切られた実例から追加した
    "やっぱ", "やっぱり", "たぶん", "きっと", "すごく", "めっちゃ", "あんまり",
    "かなり", "けっこう", "結構", "たしか", "なんとなく",
    # 連体詞（後ろに来る体言がまだ来ていない）
    "こんな", "そんな", "あんな", "どんな",
)


def looks_incomplete(text: str) -> bool:
    """発話がまだ続きそうかを、末尾の表現だけから判定する。

    音声認識の中間結果は句読点を持たないため、文の完結は語尾でしか見分けられない。
    真なら終端判定の待ちを長い方に切り替える。
    """
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith(CONTINUATION_ENDINGS)


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

    def __init__(self, mapping: Dict[str, str], swap_ratio: float = 0.33) -> None:
        if not 0.0 <= swap_ratio <= 1.0:
            raise ValueError(f"swap_ratio は 0.0〜1.0 で指定してください: {swap_ratio}")
        fixed_points = sorted(src for src, dst in mapping.items() if src == dst)
        if fixed_points:
            # 自分自身へ写る子音があると、そこだけ加工が効かず素のまま鳴ってしまう
            raise ValueError(f"置換表に自分自身へ写る子音があります: {fixed_points}")
        self.mapping: Dict[str, str] = dict(mapping)
        self.swap_ratio: float = swap_ratio

    @classmethod
    def articulatory(cls, swap_ratio: float = 0.33) -> "ConsonantSubstitutionTable":
        """調音様式・有声性を保ったまま調音点だけを変える置換表（k→t、n→m など）。

        「別の言語」ではなく「滑舌が悪い」側に寄るため、意味を壊しつつ聞き疲れしにくい。
既定は swap_ratio=0.33。0.5 → 0.67 → 0.5 と動かしたのち、**初めて聞く人には聞き取れない**という指摘で 0.33 まで下げた。開発者は応答の中身も置換規則も知っているため復元できてしまい、値を高く見積もる（経緯は DECISIONS.md）。
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
