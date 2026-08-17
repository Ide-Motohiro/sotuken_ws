"""読点で単語ごとに区切る発話（ぶっきらぼうな片言）が、疑似言語音の了解性を上げるかを聞き比べる。

背景：応答を短くするプロンプトを設計する過程で、「読点を多めに打って単語単位で区切れば、
疑似言語音でも1語ずつ聞き取りやすくなるのではないか」という仮説が出た。

この仮説には2つの効果が同時に働くと考えられ、どちらが勝つかは耳で確かめるしかない。

  (+) 区切りのポーズが語境界の手がかりになり、1語ずつ処理する時間ができる
  (-) ポーズの分だけ再生時間が延びる。同じモーラ数でも聞かされる時間が長くなり、
      認知的な負担は増える（実測では読点を増やすと秒/モーラが約2倍になる）

同じ内容を「読点あり」「読点なし」で用意し、どちらも同じ子音置換をかけて交互に鳴らす。
置換の規則は本線と共有している（domain.models.ConsonantSubstitutionTable）。

再生順は 読点なし → 読点あり → 加工なし（答え）。答えのテキストは最後まで伏せてある。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.compare_chunking
    python -m google_stt.compare_chunking --swap-ratio 0.33
    python -m google_stt.compare_chunking --no-play   # 再生せず wav 保存だけ
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
from google_stt.domain.models import ConsonantSubstitutionTable
from google_stt.infrastructure.phoneme_swap import substitute_consonants

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "chunking_samples")

# (識別子, 文脈となるユーザー発話, 読点なし, 読点あり)
# 内容と語順は揃えてあり、違いは読点の有無だけ。
SAMPLES = [
    ("weather", "今日はいい天気だね", "うん、きもちいいね", "うん、きもち、いい、ね"),
    ("weekend", "週末どこか行く？", "あー、まだ決めてないかな", "あー、まだ、決めてない、かな"),
    ("congrats", "テスト受かったよ", "おお、やったね！", "おお、やった、ね！"),
    ("forgot", "この前の話、覚えてる？", "えー、どの話だっけ", "えー、どの、話、だっけ"),
    ("refuse", "これ食べていい？", "だめ、それわたしのだよ", "だめ、それ、わたしの、だよ"),
]

PAUSE_SEC = 0.7


def measure(query: dict) -> tuple[int, float]:
    """audio_query から (モーラ数, 再生秒数) を得る。ポーズもモーラ長として積む"""
    moras = 0
    seconds = query["prePhonemeLength"] + query["postPhonemeLength"]
    for phrase in query["accent_phrases"]:
        for mora in phrase["moras"]:
            moras += 1
            seconds += (mora["consonant_length"] or 0.0) + mora["vowel_length"]
        pause_mora = phrase.get("pause_mora")
        if pause_mora:
            seconds += (pause_mora["consonant_length"] or 0.0) + pause_mora["vowel_length"]
    return moras, seconds / query["speedScale"]


def save_wav(arr: np.ndarray, sample_rate: int, filename: str) -> None:
    with open(os.path.join(OUTPUT_DIR, filename), "wb") as f:
        f.write(array_to_wav_bytes(arr.astype(np.float32), sample_rate))


def render(url: str, speaker_id: int, text: str, table: ConsonantSubstitutionTable):
    """加工なしと加工済みの両方を合成して返す"""
    base_query = fetch_query(url, text, speaker_id)
    swapped_query = substitute_consonants(copy.deepcopy(base_query), table)
    # 韻律とリズムの保持がこの方式の核心なので、破っていたら止める
    assert_prosody_preserved(base_query, swapped_query)

    base_audio, sample_rate = wav_bytes_to_array(
        synthesize(url, base_query, speaker_id, "加工なし"))
    swapped_audio, _ = wav_bytes_to_array(
        synthesize(url, swapped_query, speaker_id, "子音置換"))
    return base_query, base_audio, swapped_query, swapped_audio, sample_rate


def run(url: str, speaker_id: int, should_play: bool, swap_ratio: float) -> None:
    check_voicevox_running(url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table = ConsonantSubstitutionTable.articulatory(swap_ratio=swap_ratio)

    print(f"VOICEVOX: {url} / speaker={speaker_id} / 差し替え率={swap_ratio}")
    print(f"保存先: {OUTPUT_DIR}\n")
    print("=" * 78)
    print("聞き方：同じ内容を「読点なし」「読点あり」の順に鳴らす。")
    print("　　　　どちらが語として拾えるか、どちらが聞いていて楽かを別々に見る。")
    print("　　　　答えのテキストは最後まで伏せてある。")
    print("=" * 78)

    totals = {"読点なし": [], "読点あり": []}

    for key, context_text, flowing_text, chunked_text in SAMPLES:
        print(f"\n{'=' * 78}")
        print(f"  文脈（あなた）: 「{context_text}」")
        print("=" * 78)

        for label, text, suffix in (("読点なし", flowing_text, "flowing"),
                                    ("読点あり", chunked_text, "chunked")):
            base_query, base_audio, swapped_query, swapped_audio, sample_rate = render(
                url, speaker_id, text, table)
            moras, seconds = measure(base_query)
            totals[label].append((moras, seconds))

            save_wav(swapped_audio, sample_rate, f"{key}_{suffix}_swapped.wav")
            save_wav(base_audio, sample_rate, f"{key}_{suffix}_original.wav")

            print(f"\n  【{label}】{moras}モーラ / {seconds:.2f}秒 "
                  f"（{seconds / moras:.3f}秒/モーラ）")
            print(f"      読み: {describe_phonemes(swapped_query)}")
            if should_play:
                play(swapped_audio, sample_rate)
                time.sleep(PAUSE_SEC)

        print(f"\n  答え: 読点なし「{flowing_text}」/ 読点あり「{chunked_text}」")

    print(f"\n{'=' * 78}")
    print("まとめ")
    print("=" * 78)
    print(f"  {'':<10} {'平均モーラ':>10} {'平均秒':>8} {'秒/モーラ':>10}")
    for label, results in totals.items():
        moras = [m for m, _ in results]
        seconds = [s for _, s in results]
        mean_moras = sum(moras) / len(moras)
        mean_seconds = sum(seconds) / len(seconds)
        print(f"  {label:<10} {mean_moras:>10.1f} {mean_seconds:>8.2f} "
              f"{mean_seconds / mean_moras:>10.3f}")
    print()
    print("判断すること：")
    print("  ・読点ありの方が語として拾えるか（仮説が正しいか）")
    print("  ・拾えるとして、延びた秒数に見合うか（了解性と認知負荷のどちらを取るか）")
    print("  ・読点なしでも足りているなら、短い方を選べる")
    if not should_play:
        print("\n（--no-play のため再生はしていない）")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="読点による区切りが疑似言語音の了解性を上げるかを聞き比べる")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument("--swap-ratio", type=float, default=0.5, help="子音の差し替え率（既定0.5）")
    parser.add_argument("--no-play", action="store_true", help="再生せず wav の保存だけを行う")
    args = parser.parse_args()

    run(args.url, args.speaker, should_play=not args.no_play, swap_ratio=args.swap_ratio)


if __name__ == "__main__":
    main()
