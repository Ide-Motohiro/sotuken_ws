"""audio_query の子音を差し替えて疑似言語音を出力する TextToSpeech 実装。

合成後の波形を加工するのではなく、合成前の audio_query の中身を書き換える。各モーラの
consonant だけを差し替え、vowel・pitch・consonant_length・vowel_length・pause_mora は
一切触らない。これにより：

  - 合成するのは VOICEVOX 自身なので音質は素のまま（ノイズが混ざらない）
  - ピッチ曲線が元のままなので、疑問文は疑問文のイントネーションで鳴る
  - モーラ数・長さが元のままなので、短い相槌は短く、長い応答は長く鳴る
  - 語彙的な意味だけが消える
  - 後段のDSPが無いので追加レイテンシはゼロ（HTTPリクエスト数も VoiceVoxTTS と同じ）

置換の規則そのもの（どの子音をどの子音へ写すか、どのモーラを差し替えるか）は
ドメインの ConsonantSubstitutionTable が持つ。ここは audio_query の構造を歩いて
その規則を適用する役だけを担う。

聴取比較（compare_intelligibility_grid.py）の結果に基づき、既定は
「調音的に近い置換 × 1モーラおき」にしてある。了解性を調整したい場合は
swap_ratio を変える（1.0 に近づけるほど分かりにくくなる）。
"""
from typing import Any, Callable, Dict, Optional

from google_stt.domain.models import ConsonantSubstitutionTable
from google_stt.infrastructure.voicevox import SynthesisTiming, VoiceVoxTTS


def substitute_consonants(
    query: Dict[str, Any], table: ConsonantSubstitutionTable
) -> Dict[str, Any]:
    """audio_query の子音を置換表に従って差し替える。渡されたクエリを直接書き換える。

    母音のみのモーラと「ン」は consonant が None になっている。子音を持たないところに
    子音を捏造すると consonant_length が無いまま発話長が変わるため、触らない。
    これらは差し替え対象の通し番号にも数えない。
    """
    index = 0
    for accent_phrase in query["accent_phrases"]:
        for mora in accent_phrase["moras"]:
            consonant = mora["consonant"]
            if consonant is None:
                continue
            should_swap = table.should_swap(index)
            index += 1
            if not should_swap:
                continue
            replacement = table.substitute(consonant)
            # 置換表に無い子音は素通しする。VOICEVOX が新しい音素表記を返すように
            # なった場合にここで黙って落ちるより、素の音で鳴った方が気付ける。
            if replacement is not None:
                mora["consonant"] = replacement
    return query


class PhonemeSwapTTS(VoiceVoxTTS):
    """VOICEVOX の合成前クエリの子音を差し替えて再生する疑似言語音 TTS。

    VoiceVoxTTS の派生なので、再生・エコー防止用の is_speaking の扱いは親と共通。
    文章全体からクエリを作る必要があるため、speak_stream は親と同じく
    ストリームを結合してから合成する（レイテンシ隠蔽の効果は無い）。
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:50021",
        speaker_id: int = 3,
        substitution_table: Optional[ConsonantSubstitutionTable] = None,
        on_timing: Optional[Callable[[SynthesisTiming], None]] = None,
    ) -> None:
        super().__init__(url=url, speaker_id=speaker_id, on_timing=on_timing)
        self.substitution_table: ConsonantSubstitutionTable = (
            substitution_table
            if substitution_table is not None
            else ConsonantSubstitutionTable.articulatory(swap_ratio=0.5)
        )

    def _transform_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return substitute_consonants(query, self.substitution_table)
