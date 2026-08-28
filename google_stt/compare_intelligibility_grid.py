"""疑似言語音の了解性を「発話長・語種 × 加工強度」の格子で聞き比べる比較スクリプト。

compare_blur / compare_phoneme_swap は加工強度だけを振っていて、サンプル文も長文・
文脈なしだった。これは了解性にとって最悪条件で、そこだけを見て「分からなさすぎる」と
判断していた可能性がある。このスクリプトは軸を2本に増やす：

  軸1（発話長・語種）  感動詞/オノマトペ → 短い定型応答 → 片言 → 通常の文
  軸2（加工強度）      置換表（調音的に近い／遠い） × 差し替え率

加工は audio_query の consonant 差し替えのみで、母音・pitch・長さ・ポーズは触らない
（合成するのは VOICEVOX 自身なので音質は素のまま）。

**置換の規則は本線と共有している**。置換表は domain.models.ConsonantSubstitutionTable、
適用は infrastructure.phoneme_swap.substitute_consonants で、PhonemeSwapTTS が本番で
使うものと同一。ここで耳で決めた設定がそのまま本線の設定になる。

compare_phoneme_swap.py との違いは2点：

  1. **置換表が固定**。あちらはモーラごとに rng.choice していたので、同じ「か」でも
     出るたび別の音になっていた。それは言語ではなく喃語として知覚される。
  2. **調音的な近さで置換先を選べる**。無声破裂音は無声破裂音へ（k→t）、鼻音は鼻音へ
     （n→m）のように調音様式・有声性を保ったまま調音点だけ変えると、「別の言語」ではなく
     「滑舌が悪い」側に寄る。

文脈の扱い：各サンプルは（ユーザー発話, エージェント応答）の対で、ユーザー発話は画面に
文字で出すだけで音は鳴らさない。実際の対話では自分が何を言ったかは分かっていて相手の
応答だけを聞く、という状況を再現するため。

再生順は加工が強い方から弱い方へ。応答テキストは最後（加工なし）まで伏せてある。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.compare_intelligibility_grid
    python -m google_stt.compare_intelligibility_grid --ratios 1.0 0.67 0.5 0.33
    python -m google_stt.compare_intelligibility_grid --no-play   # 再生せず wav 保存だけ
"""
import argparse
import copy
import os
import time

import numpy as np

from google_stt.build_mora_bank import array_to_wav_bytes, wav_bytes_to_array
from google_stt.compare_phoneme_swap import (
    DEFAULT_SPEAKER_ID,
    DEFAULT_VOICEVOX_URL,
    assert_prosody_preserved,
    check_voicevox_running,
    describe_phonemes,
    fetch_query,
    play,
    synthesize,
)
from google_stt.domain.models import ARTICULATION_GROUPS, CONSONANTS, ConsonantSubstitutionTable
from google_stt.infrastructure.phoneme_swap import substitute_consonants

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "grid_samples")

# (ユーザー発話, エージェント応答) の対。ユーザー発話は文脈として表示するだけで鳴らさない。
# 「片言」と「通常の文」は同じ文脈を共有させてあるので、発話長だけの差を直接比較できる。
# interj_2「うん」は母音と撥音だけで子音を持つモーラが無く、この方式では原理的に加工されない。
# 「感動詞は壊れにくい」を極端な形で示す例として意図的に残してある。
SAMPLES = [
    ("interj_1", "感動詞・オノマトペ", "それ、昨日ついに完成したんだ", "わー！"),
    ("interj_2", "感動詞・オノマトペ", "ちょっと手伝ってくれない？", "うん"),
    ("interj_3", "感動詞・オノマトペ", "テスト受かったよ", "やったー！"),
    ("short_1", "短い定型応答", "明日、雨降るかな", "ふりそう"),
    ("short_2", "短い定型応答", "これ、食べていい？", "だめだよ"),
    ("broken_1", "片言", "週末どこか行った？", "こうえん、たのしかった"),
    ("broken_2", "片言", "お腹すいたなあ", "ぼくも、なにか、たべたい"),
    ("full_1", "通常の文（対照）", "週末どこか行った？",
     "近所の公園まで散歩に行って、それから本屋に寄って帰ってきたよ"),
    ("full_2", "通常の文（対照）", "お腹すいたなあ",
     "そろそろお昼の時間だし、何か作って食べた方がいいと思うよ"),
]

DEFAULT_RATIOS = [1.0, 0.33]
PAUSE_SEC = 0.7


def build_levels(ratios: list[float]) -> list[tuple[str, str, ConsonantSubstitutionTable]]:
    """再生する条件を加工が強い順に組み立てる。

    先頭は比較対照の「遠い置換・全モーラ」（従来の swap_consonant に相当）。
    その後に「近い置換」を差し替え率の高い順（＝分かりにくい順）に並べる。
    """
    levels = [(
        "distant100",
        "遠い置換・全モーラ（比較対照。従来の swap_consonant に相当）",
        ConsonantSubstitutionTable.distant(swap_ratio=1.0),
    )]
    for ratio in sorted(ratios, reverse=True):
        levels.append((
            f"near{int(round(ratio * 100)):03d}",
            f"近い置換・差し替え率 {ratio:.2f}",
            ConsonantSubstitutionTable.articulatory(swap_ratio=ratio),
        ))
    return levels


def count_swaps(before: dict, after: dict) -> tuple[int, int]:
    """(実際に差し替わったモーラ数, 子音を持つモーラ数) を実際の差分から数える。

    差し替え規則を再実装せず結果を突き合わせるので、本線の実装と食い違いようがない。
    """
    consonant_moras = 0
    swapped = 0
    for before_phrase, after_phrase in zip(before["accent_phrases"], after["accent_phrases"]):
        for before_mora, after_mora in zip(before_phrase["moras"], after_phrase["moras"]):
            if before_mora["consonant"] is None:
                continue
            consonant_moras += 1
            if before_mora["consonant"] != after_mora["consonant"]:
                swapped += 1
    return swapped, consonant_moras


def find_unmapped_consonants(query: dict, table: ConsonantSubstitutionTable) -> list[str]:
    """置換表に無い子音を洗い出す。VOICEVOX が未知の音素表記を返し始めたら気付けるように"""
    present = {
        mora["consonant"]
        for phrase in query["accent_phrases"]
        for mora in phrase["moras"]
        if mora["consonant"] is not None
    }
    return sorted(present - set(table.mapping))


def save_wav(arr: np.ndarray, sample_rate: int, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(array_to_wav_bytes(arr.astype(np.float32), sample_rate))


def print_tables() -> None:
    articulatory = ConsonantSubstitutionTable.articulatory()
    distant = ConsonantSubstitutionTable.distant()

    print("=" * 78)
    print("固定置換表（全発話で一貫して使う。同じ子音は常に同じ子音へ写る）")
    print("=" * 78)
    print("  近い置換（調音様式・有声性を保ち、調音点だけ変える）")
    for group in ARTICULATION_GROUPS:
        print("      " + "  ".join(f"{c}->{articulatory.substitute(c)}" for c in group))
    print("\n  遠い置換（調音的な近さを無視して環を半周ずらす）")
    pairs = [f"{c}->{distant.substitute(c)}" for c in CONSONANTS]
    for i in range(0, len(pairs), 8):
        print("      " + "  ".join(pairs[i:i + 8]))
    print()


def run(url: str, speaker_id: int, should_play: bool, ratios: list[float]) -> None:
    check_voicevox_running(url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"VOICEVOX: {url} / speaker={speaker_id}（speedScale等は既定値のまま）")
    print(f"保存先: {OUTPUT_DIR}\n")

    print_tables()
    levels = build_levels(ratios)

    print("=" * 78)
    print("聞き方：文脈（あなたの発話）は先に表示される。応答テキストは最後まで伏せてある。")
    print("　　　　加工が強い順に鳴るので、どの水準で意味が立ち上がるかを確かめる。")
    print("=" * 78)

    for sample_key, category, context_text, reply_text in SAMPLES:
        base_query = fetch_query(url, reply_text, speaker_id)
        base_bytes = synthesize(url, base_query, speaker_id, "加工なし")
        base_audio, sample_rate = wav_bytes_to_array(base_bytes)
        save_wav(base_audio, sample_rate, f"{sample_key}_original.wav")

        mora_count = sum(len(p["moras"]) for p in base_query["accent_phrases"])
        print(f"\n{'=' * 78}")
        print(f"［{category}］{sample_key}　{mora_count}モーラ / "
              f"{len(base_audio) / sample_rate:.2f}秒")
        print(f"  文脈（あなた）: 「{context_text}」")
        print(f"{'=' * 78}")

        base_reading = describe_phonemes(base_query)
        # 同じ読みになった条件を鳴らし直しても聞き比べにならないので、既出の読みを控えておく
        played_readings: dict[str, str] = {base_reading: "加工なし"}

        for level_key, level_label, table in levels:
            unmapped = find_unmapped_consonants(base_query, table)
            if unmapped:
                print(f"  ⚠ 置換表に無い子音があり、素のまま鳴ります: {', '.join(unmapped)}")

            swapped_query = substitute_consonants(copy.deepcopy(base_query), table)
            # この方式の核心は韻律とリズムの保持なので、破っていたら止める
            assert_prosody_preserved(base_query, swapped_query)
            swapped_count, consonant_moras = count_swaps(base_query, swapped_query)

            swapped_bytes = synthesize(url, swapped_query, speaker_id, level_label)
            swapped_audio, swapped_rate = wav_bytes_to_array(swapped_bytes)
            save_wav(swapped_audio, swapped_rate, f"{sample_key}_{level_key}.wav")

            reading = describe_phonemes(swapped_query)
            print(f"\n  {level_label}")
            print(f"      読み: {reading}　（差し替え {swapped_count}/{consonant_moras} モーラ）")

            if consonant_moras == 0:
                print("      ※ 子音を持つモーラが無いため、この発話はこの方式では加工できない")
            duplicate_of = played_readings.get(reading)
            if duplicate_of is not None:
                print(f"      ※「{duplicate_of}」と同一の読みなので再生を省略する")
                continue
            played_readings[reading] = level_label.split("（")[0].strip()

            if should_play:
                play(swapped_audio, swapped_rate)
                time.sleep(PAUSE_SEC)

        print(f"\n  加工なし（＝答え）: 「{reply_text}」")
        print(f"      読み: {base_reading}")
        if should_play:
            play(base_audio, sample_rate)
            time.sleep(PAUSE_SEC)

    print(f"\n{'=' * 78}")
    print("完了。保存済みの wav で何度でも聞き直せる:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("見るべき点：")
    print("  ・同じ加工強度でも、発話が短いほど分かるか（軸1が効いているか）")
    print("  ・感動詞は音素が壊れても韻律で意味が残るか（語種が効いているか）")
    print("  ・「近い置換」が「遠い置換」より、別言語ではなく滑舌の悪さに聞こえるか")
    print("  ・broken_* と full_* は文脈が同じなので、発話長だけの差を直接比べられる")
    print()
    print("良かった差し替え率は本線にそのまま渡せる:")
    print("  python -m google_stt.main_phoneme_swap --swap-ratio 0.33")
    if not should_play:
        print("\n（--no-play のため再生はしていない）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="発話長・語種 × 加工強度の格子で疑似言語音の了解性を聞き比べる")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
        help=f"「近い置換」で試す差し替え率（0.0〜1.0、複数可）。既定は {DEFAULT_RATIOS}",
    )
    parser.add_argument("--no-play", action="store_true", help="再生せず wav の保存だけを行う")
    args = parser.parse_args()

    run(args.url, args.speaker, should_play=not args.no_play, ratios=args.ratios)


if __name__ == "__main__":
    main()
