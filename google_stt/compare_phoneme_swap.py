"""audio_query の音素を差し替えて疑似言語音を作る方式の検証スクリプト。

合成後の波形を加工するのではなく、合成前の audio_query の中身を書き換える。
各 mora が持つ consonant / vowel だけを差し替え、pitch・consonant_length・
vowel_length・pause_mora は一切触らない。これにより：

  - 合成するのは VOICEVOX 自身なので音質は素のまま（ノイズが混ざらない）
  - ピッチ曲線が元のままなので、疑問文は疑問文のイントネーションで鳴る
  - モーラ数・長さが元のままなので、短い相槌は短く、長い応答は長く鳴る
  - 語彙的な意味だけが消える
  - 後段のDSPが無いので追加レイテンシはゼロ（HTTPリクエストは現状と同じ1回）

比較する4条件：
  (a) 加工なし（基準音）
  (b) 子音のみランダム化（母音の並びは元のまま＝最も「あと一歩で分かりそう」）
  (c) 子音・母音とも限定セットに差し替え（Animalese寄り）
  (d) 子音・母音とも全ランダム

差し替えは音素の種類だけで決まる静的な規則で、入力テキストの内容では分岐しない。
ランダム性は一様分布からの選択のみ。既定では固定シードで再現性を持たせている。

音素表記は VOICEVOX に総当たりで問い合わせて確認した実測値（下の定数のコメント参照）。

サンプルレートについて：VOICEVOX の出力は 24000Hz。再生・保存はすべて合成結果の wav から
読み取った実際のサンプルレートを使う（PseudoVoiceTTS の 44100Hz とは無関係）。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.compare_phoneme_swap
    python -m google_stt.compare_phoneme_swap --no-play   # 再生せず wav 保存だけ
    python -m google_stt.compare_phoneme_swap --seed 7    # 別のランダム割り当てを試す
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import requests
import sounddevice as sd

from google_stt.build_mora_bank import array_to_wav_bytes, wav_bytes_to_array
from google_stt.infrastructure.voicevox import VoiceVoxTTS

# 話者などの既定値は本番実装（infrastructure/voicevox.py）から借りる。
# VoiceVoxTTS のコンストラクタは値を保持するだけで副作用はない。
_PRODUCTION_DEFAULTS = VoiceVoxTTS()
DEFAULT_SPEAKER_ID = _PRODUCTION_DEFAULTS.speaker_id

# 本番実装が使っている "localhost" はこの環境では ::1（IPv6）を先に返し、VOICEVOX が
# IPv6 で待ち受けていないため、TCP接続のたびに約2秒のフォールバック待ちが入る
# （measure_voicevox_latency.py で実測済み）。待たされるだけなので 127.0.0.1 を既定にする。
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "phoneme_samples")

# VOICEVOX に総当たりで投げて synthesis が通ることを確認した子音（speaker=3 で実測）。
# "ng" と "q" は 500 を返したため除外している。
CONSONANTS = [
    "k", "ky", "g", "gy", "s", "sh", "z", "j", "t", "ch", "ts", "ty",
    "d", "dy", "n", "ny", "h", "hy", "f", "b", "by", "p", "py", "m", "my",
    "y", "r", "ry", "w", "v",
]

# 差し替え対象にする母音。これ以外（N=撥音 / cl=促音 / pau=ポーズ / U,I=無声化母音）は
# リズムや無声区間そのものなので保持する。無声化母音は pitch=0.0 が入っており、
# 有声母音に変えると触れない制約の pitch と矛盾するため対象外とする。
VOWELS = ["a", "i", "u", "e", "o"]
PRESERVED_VOWELS = {"N", "cl", "pau", "U", "I"}

# (c) 限定セット。少数の音素だけを使い、より分かりにくくする
LIMITED_CONSONANTS = ["p", "t", "k", "m"]
LIMITED_VOWELS = ["a", "i", "u"]

SAMPLE_TEXTS = [
    ("aizuchi", "相槌", "なるほどね"),
    ("statement", "平叙文", "今日はとてもいい天気だから散歩に行きたいと思う。"),
    ("question", "疑問文", "明日の午後から一緒に映画を見に行きませんか？"),
]

PAUSE_SEC = 0.8


def check_voicevox_running(url: str) -> None:
    """VOICEVOXに到達できなければ、その旨を明示して異常終了する"""
    try:
        response = requests.get(f"{url}/speakers", timeout=3.0)
    except requests.exceptions.RequestException as e:
        print(
            f"VOICEVOXに接続できません（{url}）。\n"
            f"VOICEVOXを起動してから再実行してください。\n"
            f"  詳細: {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    # 起動しているのに異常応答を返す場合は握りつぶさず例外を上げる
    response.raise_for_status()


def fetch_query(url: str, text: str, speaker_id: int) -> dict:
    """audio_query を取得する。speedScale等のトップレベルパラメータは一切変更しない"""
    response = requests.post(
        f"{url}/audio_query",
        params={"text": text, "speaker": speaker_id},
    )
    response.raise_for_status()
    return response.json()


def synthesize(url: str, query: dict, speaker_id: int, condition: str) -> bytes:
    """クエリから音声を合成する。差し替えた音素が拒否された場合は内容を添えて失敗させる"""
    response = requests.post(
        f"{url}/synthesis",
        params={"speaker": speaker_id},
        json=query,
    )
    if response.status_code != 200:
        phonemes = describe_phonemes(query)
        raise RuntimeError(
            f"synthesis が条件「{condition}」で失敗した（HTTP {response.status_code}）。\n"
            f"  応答: {response.text[:300]}\n"
            f"  差し替え後の音素列: {phonemes}\n"
            f"  VOICEVOX が受け付けない音素表記が含まれている可能性がある。"
        )
    return response.content


# --- 手順1：audio_query の構造調査 ---

def inspect_query_structure(url: str, speaker_id: int) -> None:
    """audio_query の accent_phrases / moras の中身をそのまま表示する"""
    text = "こんにちは、元気ですか？"
    query = fetch_query(url, text, speaker_id)

    print("=" * 78)
    print(f"【手順1】audio_query の構造調査　テキスト: {text}")
    print("=" * 78)

    print("\n-- トップレベル（今回は一切変更しない） --")
    for key, value in query.items():
        if key != "accent_phrases":
            print(f"  {key}: {value!r}")

    print("\n-- accent_phrases --")
    for i, phrase in enumerate(query["accent_phrases"]):
        print(f"  [accent_phrase {i}] accent={phrase['accent']} "
              f"is_interrogative={phrase.get('is_interrogative')}")
        for j, mora in enumerate(phrase["moras"]):
            print(f"      mora[{j}] {json.dumps(mora, ensure_ascii=False)}")
        print(f"      pause_mora: {json.dumps(phrase.get('pause_mora'), ensure_ascii=False)}")

    print("\n-- この構造から決めた差し替え規則 --")
    print("  consonant: 'k' 'ch' 'ky' のようなローマ字表記。母音のみのモーラと「ン」は null。")
    print("             → 元が null のモーラには子音を入れない（consonant_length の捏造になるため）")
    print(f"  vowel    : {VOWELS} のほか N（撥音）/ cl（促音）/ pau（ポーズ）/ U,I（無声化母音）。")
    print(f"             → 差し替えるのは {VOWELS} のみ。{sorted(PRESERVED_VOWELS)} は保持する")
    print("  pitch / consonant_length / vowel_length / pause_mora は一切変更しない")
    print()


# --- 音素の差し替え ---

def swap_phonemes(query: dict, mode: str, rng: np.random.Generator) -> dict:
    """音素を差し替えた新しいクエリを返す。元のクエリは変更しない。

    mode:
      "consonant" 子音のみを別の子音へ（母音は保持）
      "limited"   子音・母音とも限定セットから選ぶ
      "random"    子音・母音とも全候補から一様に選ぶ
    """
    swapped = copy.deepcopy(query)

    if mode == "consonant":
        consonant_pool, vowel_pool = CONSONANTS, None
    elif mode == "limited":
        consonant_pool, vowel_pool = LIMITED_CONSONANTS, LIMITED_VOWELS
    elif mode == "random":
        consonant_pool, vowel_pool = CONSONANTS, VOWELS
    else:
        raise ValueError(f"未知のモード: {mode}")

    for phrase in swapped["accent_phrases"]:
        for mora in phrase["moras"]:
            # 元から子音を持つモーラだけを対象にする（長さを維持するため）
            if mora["consonant"] is not None:
                candidates = [c for c in consonant_pool if c != mora["consonant"]]
                if not candidates:  # 限定セットが元の子音1種しか含まない場合の保険
                    candidates = consonant_pool
                mora["consonant"] = str(rng.choice(candidates))

            # 撥音・促音・ポーズ・無声化母音は構造そのものなので触らない
            if vowel_pool is not None and mora["vowel"] in VOWELS:
                mora["vowel"] = str(rng.choice(vowel_pool))

    return swapped


def assert_prosody_preserved(original: dict, swapped: dict) -> None:
    """pitch・長さ・ポーズが1つも変わっていないことを確認する。
    この方式の核心が「韻律とリズムの保持」なので、破っていたら止める。"""
    original_phrases = original["accent_phrases"]
    swapped_phrases = swapped["accent_phrases"]
    if len(original_phrases) != len(swapped_phrases):
        raise RuntimeError("accent_phrase の数が変わっている")

    for i, (original_phrase, swapped_phrase) in enumerate(zip(original_phrases, swapped_phrases)):
        if json.dumps(original_phrase.get("pause_mora"), sort_keys=True) != \
                json.dumps(swapped_phrase.get("pause_mora"), sort_keys=True):
            raise RuntimeError(f"accent_phrase[{i}] の pause_mora が変わっている")
        if original_phrase["accent"] != swapped_phrase["accent"]:
            raise RuntimeError(f"accent_phrase[{i}] の accent が変わっている")
        # 疑問文の語尾上昇を担うフラグなので、疑問文条件の検証のために必ず保持する
        if original_phrase.get("is_interrogative") != swapped_phrase.get("is_interrogative"):
            raise RuntimeError(f"accent_phrase[{i}] の is_interrogative が変わっている")

        original_moras = original_phrase["moras"]
        swapped_moras = swapped_phrase["moras"]
        if len(original_moras) != len(swapped_moras):
            raise RuntimeError(f"accent_phrase[{i}] のモーラ数が変わっている")

        for j, (original_mora, swapped_mora) in enumerate(zip(original_moras, swapped_moras)):
            for key in ("pitch", "consonant_length", "vowel_length"):
                if original_mora[key] != swapped_mora[key]:
                    raise RuntimeError(
                        f"accent_phrase[{i}].mora[{j}] の {key} が変わっている "
                        f"({original_mora[key]!r} -> {swapped_mora[key]!r})"
                    )

    for key, value in original.items():
        if key != "accent_phrases" and swapped[key] != value:
            raise RuntimeError(f"トップレベルの {key} が変わっている")


def describe_phonemes(query: dict) -> str:
    """差し替え後の読みをローマ字で組み立てて表示用に返す"""
    parts = []
    for phrase in query["accent_phrases"]:
        for mora in phrase["moras"]:
            vowel = mora["vowel"]
            if vowel == "N":
                parts.append("n")
            elif vowel == "cl":
                parts.append("Q")
            elif vowel == "pau":
                parts.append("_")
            else:
                consonant = mora["consonant"] or ""
                parts.append(consonant + vowel)
        if phrase.get("pause_mora"):
            parts.append("_")
    return " ".join(parts)


# --- 再生・保存 ---

def save_wav(arr: np.ndarray, sample_rate: int, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(array_to_wav_bytes(arr.astype(np.float32), sample_rate))
    return path


def play(arr: np.ndarray, sample_rate: int) -> None:
    # 合成結果の実サンプルレート（VOICEVOXは24000Hz）で鳴らす。ここを間違えると全条件が早回しになる
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out_stream:
        out_stream.write(arr.astype(np.float32))


def announce_and_play(label: str, reading: str, arr: np.ndarray,
                      sample_rate: int, should_play: bool) -> None:
    print(f"  {label}（{len(arr) / sample_rate:.2f}秒）")
    print(f"      読み: {reading}")
    if should_play:
        play(arr, sample_rate)
        time.sleep(PAUSE_SEC)


def run(url: str, speaker_id: int, should_play: bool, seed: int) -> None:
    check_voicevox_running(url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"VOICEVOX: {url} / speaker={speaker_id}（pitchScale等は既定値のまま）")
    print(f"保存先: {OUTPUT_DIR}\n")

    inspect_query_structure(url, speaker_id)

    rng = np.random.default_rng(seed)

    conditions = [
        ("original", "(a) 加工なし（基準音）", None),
        ("swap_consonant", "(b) 子音のみランダム化（母音は元のまま）", "consonant"),
        ("swap_limited", f"(c) 限定セット 子音{LIMITED_CONSONANTS} 母音{LIMITED_VOWELS}", "limited"),
        ("swap_random", "(d) 子音・母音とも全ランダム", "random"),
    ]

    print("=" * 78)
    print("【手順2】音素を差し替えて合成・再生")
    print("=" * 78)

    for text_key, text_label, text in SAMPLE_TEXTS:
        base_query = fetch_query(url, text, speaker_id)
        base_audio_bytes = synthesize(url, base_query, speaker_id, "加工なし")
        base_audio, sample_rate = wav_bytes_to_array(base_audio_bytes)
        base_reading = describe_phonemes(base_query)
        save_wav(base_audio, sample_rate, f"original_{text_key}.wav")

        print(f"\n{'=' * 78}")
        print(f"サンプル文［{text_label}］: {text}（{len(text)}文字 / {sample_rate}Hz）")
        print("=" * 78)

        for file_key, condition_label, mode in conditions:
            if mode is None:
                continue

            swapped_query = swap_phonemes(base_query, mode, rng)
            assert_prosody_preserved(base_query, swapped_query)

            swapped_bytes = synthesize(url, swapped_query, speaker_id, condition_label)
            swapped_audio, swapped_rate = wav_bytes_to_array(swapped_bytes)
            save_wav(swapped_audio, swapped_rate, f"{file_key}_{text_key}.wav")

            # 各条件の前に基準音を挟んで耳をリセットする
            print("\n--- 基準音 ---")
            announce_and_play("(a) 加工なし", base_reading, base_audio, sample_rate, should_play)
            print(f"--- {condition_label} ---")
            announce_and_play(condition_label, describe_phonemes(swapped_query),
                              swapped_audio, swapped_rate, should_play)

    print("\n完了。保存済みの wav で何度でも聞き直せる:")
    print(f"  {OUTPUT_DIR}")
    if not should_play:
        print("（--no-play のため再生はしていない）")


def main() -> None:
    parser = argparse.ArgumentParser(description="audio_queryの音素差し替えによる疑似言語音の検証")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument("--seed", type=int, default=0, help="音素割り当てのシード（再現性のため固定）")
    parser.add_argument("--no-play", action="store_true", help="再生せず wav の保存だけを行う")
    args = parser.parse_args()

    run(args.url, args.speaker, should_play=not args.no_play, seed=args.seed)


if __name__ == "__main__":
    main()
