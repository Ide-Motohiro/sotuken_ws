"""読点の無音を合成から外し、クライアント側で挿し戻す方式の聞き比べ（不採用の記録）。

VOICEVOX は無音も通常のモーラと同じ計算量で生成するため、読点の多い応答では
合成時間の約34%（中央値441ms）が無音の生成に費やされる。無音を長さ0にして合成し、
出来た wav に無音を挿し戻せばその分だけ速くなる、という案の検証。

4つを作って聞き比べる：
  A 素の合成（現行）
  B 無音を長さ0にして合成したまま（区切りが消えるので了解性が落ちるはず）
  C B に完全な無音（ゼロ）を挿し戻したもの
  D B にノイズフロアを挿し戻したもの（A と同じに聞こえれば採用できる）

挿入位置はクエリの長さ合計から予測する。予測は最大15msずれるため、予測位置の周辺で
振幅が最小の点に吸着させてから挿入する（母音の途中で切ると不連続点になる）。

**C は聴取で「ぶつ切り」と判定された。** 原因は挿し戻す中身がゼロだったこと。
素の合成の「無音区間」は無音ではなく、発話に対して約-30dB のノイズフロア
（RMS 23〜49、発話全体は約1000）が常に乗っている。ゼロを挿すとそれが読点のたびに
400ms前後消えるため、音が途切れて聞こえる。D はこのノイズフロアを埋め戻す。

素材は無音なし版の先頭 `prePhonemeLength` の区間から採る（RMS 19〜43 で素の無音区間と
同程度）。足りない分は往復させて並べる。往路の終端と復路の始端が同じ標本になるので、
継ぎ目で波形が飛ばない。

    python -m google_stt.compare_pause_splice
    python -m google_stt.compare_pause_splice --no-play   # 音を鳴らさず生成と計測だけ
"""
import argparse
import array
import copy
import io
import os
import statistics
import sys
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import requests
import winsound

VOICEVOX_URL = "http://127.0.0.1:50021"
SPEAKER_ID = 3
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pause_samples")

# 吸着させる窓の半幅。予測誤差（実測で最大15ms）を包める幅にしてある
SNAP_WINDOW_SEC = 0.025

# 実際に main_phoneme_swap.py が生成した応答から採った
REPLIES = [
    "うん、きもち、いいね",
    "あー、まだ、決めてない、かな",
    "おお、最近、シティポップ、良い、だよ。",
    "うん、きみに、あう、もの、探す、よ。",
]


def audio_query(text: str) -> Dict[str, Any]:
    response = requests.post(
        f"{VOICEVOX_URL}/audio_query", params={"text": text, "speaker": SPEAKER_ID})
    response.raise_for_status()
    return response.json()


def synthesize(query: Dict[str, Any]) -> Tuple[bytes, float]:
    """(wavバイト列, 所要秒) を返す"""
    start = time.perf_counter()
    response = requests.post(
        f"{VOICEVOX_URL}/synthesis", params={"speaker": SPEAKER_ID}, json=query)
    response.raise_for_status()
    return response.content, time.perf_counter() - start


def strip_pauses(query: Dict[str, Any]) -> Dict[str, Any]:
    """pause_mora の長さを 0 にした新しいクエリを返す（元は壊さない）"""
    stripped = copy.deepcopy(query)
    for phrase in stripped["accent_phrases"]:
        pause = phrase.get("pause_mora")
        if pause:
            pause["vowel_length"] = 0.0
            if pause["consonant_length"] is not None:
                pause["consonant_length"] = 0.0
    return stripped


def pause_plan(query: Dict[str, Any]) -> List[Tuple[float, float]]:
    """無音を挿し戻す位置を (無音を除いた時間軸での秒, 無音の秒) の並びで返す。

    無音の長さは 0 なので、位置の計算では無音そのものを足さない。これがそのまま
    「無音を抜いて合成した wav の中での位置」になる。
    """
    speed = query["speedScale"]
    elapsed = query["prePhonemeLength"]
    plan = []
    for phrase in query["accent_phrases"]:
        for mora in phrase["moras"]:
            elapsed += (mora["consonant_length"] or 0.0) + mora["vowel_length"]
        pause = phrase.get("pause_mora")
        if pause:
            length = (pause["consonant_length"] or 0.0) + pause["vowel_length"]
            if length > 0.0:
                plan.append((elapsed / speed, length / speed))
    return plan


def read_wav(data: bytes) -> Tuple[array.array, int, int, int]:
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"16bit 以外は想定していない: {w.getsampwidth() * 8}bit")
        frames = w.readframes(w.getnframes())
        samples = array.array("h")
        samples.frombytes(frames)
        return samples, w.getframerate(), w.getnchannels(), w.getsampwidth()


def write_wav(samples: array.array, rate: int, channels: int, width: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buffer.getvalue()


def snap_to_quiet(samples: array.array, center: int, window: int) -> int:
    """center の周辺 ±window で振幅が最小のサンプル位置を返す。

    予測位置がそのまま母音の途中に当たると、そこに無音を挟んだ瞬間に波形が
    不連続になってクリックノイズが出る。振幅がゼロに近い点まで寄せてから挿す。
    """
    low = max(0, center - window)
    high = min(len(samples), center + window + 1)
    if low >= high:
        return min(max(center, 0), len(samples))
    best = low
    best_value = abs(samples[low])
    for i in range(low + 1, high):
        value = abs(samples[i])
        if value < best_value:
            best, best_value = i, value
    return best


def tile_noise(source: array.array, count: int) -> array.array:
    """source を往復させて count 標本ぶん並べる。

    単純な繰り返しだと継ぎ目で末尾→先頭に飛んで段差ができる。往路の終端と復路の
    始端は同じ標本なので、往復させれば段差が出ない。周期性も往復のぶん目立ちにくい。
    """
    if not source:
        return array.array("h", [0]) * count
    reverse = array.array("h", reversed(source))
    filled = array.array("h")
    forward = True
    while len(filled) < count:
        filled.extend(source if forward else reverse)
        forward = not forward
    return filled[:count]


def insert_pauses(wav_bytes: bytes, plan: List[Tuple[float, float]],
                  noise_source: Optional[array.array] = None) -> Tuple[bytes, List[int]]:
    """無音を抜いて合成した wav に、無音を挿し戻す。(wav, 挿入点での振幅) を返す。

    noise_source を渡すとゼロではなくその素材で埋める。
    """
    samples, rate, channels, width = read_wav(wav_bytes)
    window = int(SNAP_WINDOW_SEC * rate)
    result = array.array("h")
    amplitudes = []
    previous = 0
    for position_sec, length_sec in plan:
        predicted = int(position_sec * rate)
        index = snap_to_quiet(samples, predicted, window)
        if index < previous:      # 吸着で順序が入れ替わったら諦めて予測位置を使う
            index = max(previous, min(predicted, len(samples)))
        amplitudes.append(abs(samples[index]) if index < len(samples) else 0)
        result.extend(samples[previous:index])
        count = int(length_sec * rate)
        if noise_source is None:
            result.extend(array.array("h", [0]) * count)
        else:
            result.extend(tile_noise(noise_source, count))
        previous = index
    result.extend(samples[previous:])
    return write_wav(result, rate, channels, width), amplitudes


def leading_noise(wav_bytes: bytes, query: Dict[str, Any]) -> array.array:
    """発話が始まる前の区間（prePhonemeLength）を取り出す。ここがノイズフロア"""
    samples, rate, _, _ = read_wav(wav_bytes)
    length = int(query["prePhonemeLength"] / query["speedScale"] * rate)
    return samples[:max(length, 1)]


def rms(samples: array.array) -> float:
    if not samples:
        return 0.0
    return (sum(float(v) * v for v in samples) / len(samples)) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description="無音を抜いて合成し挿し戻す方式の聞き比べ")
    parser.add_argument("--no-play", action="store_true", help="音を鳴らさず生成と計測だけ行う")
    parser.add_argument("--with-zero", action="store_true",
                        help="ゼロ埋め版（C）も鳴らす。ノイズフロアの有無を聞き比べる用")
    parser.add_argument("--with-bare", action="store_true",
                        help="無音なし版（B）も鳴らす。区切りが消えた状態の参考")
    args = parser.parse_args()

    try:
        requests.get(f"{VOICEVOX_URL}/version", timeout=3).raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"VOICEVOXに接続できない: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    saved = []
    all_amplitudes = []
    for number, text in enumerate(REPLIES, start=1):
        query = audio_query(text)
        plan = pause_plan(query)
        plain_wav, plain_sec = synthesize(query)
        bare_wav, bare_sec = synthesize(strip_pauses(query))
        zero_wav, amplitudes = insert_pauses(bare_wav, plan)
        noise = leading_noise(bare_wav, query)
        noise_wav, _ = insert_pauses(bare_wav, plan, noise_source=noise)
        all_amplitudes.extend(amplitudes)

        paths = {}
        for label, data in (("A_素", plain_wav), ("B_無音なし", bare_wav),
                            ("C_ゼロ埋め", zero_wav), ("D_ノイズ埋め", noise_wav)):
            path = os.path.join(OUTPUT_DIR, f"{number}_{label}.wav")
            with open(path, "wb") as f:
                f.write(data)
            paths[label] = path
        saved.append((text, paths))

        peak = max(amplitudes) if amplitudes else 0
        print(f"[{number}] 「{text}」")
        print(f"     合成 A {plain_sec*1000:>5.0f}ms → B {bare_sec*1000:>5.0f}ms "
              f"（{(plain_sec-bare_sec)*1000:>4.0f}ms 短縮）")
        print(f"     挿入点 {len(plan)}箇所 / 挿入点の振幅 最大 {peak}（32767が最大振幅）")
        print(f"     埋め戻す素材 {len(noise) / 24000 * 1000:.0f}ms / RMS {rms(noise):.1f}"
              f"（発話全体の RMS は {rms(read_wav(plain_wav)[0]):.0f}）")

    if all_amplitudes:
        print(f"\n挿入点の振幅：中央値 {statistics.median(all_amplitudes):.0f} / "
              f"最大 {max(all_amplitudes)}（32767が最大振幅）")
        print("この値が大きいほど波形の不連続が大きく、クリックノイズが出やすい。")
    print(f"\n音源: {OUTPUT_DIR}")

    if args.no_play:
        return

    print("\nA（素）→ D（ノイズ埋め）の順に鳴らす。違いが分かるかを聞く。")
    for text, paths in saved:
        print(f"\n「{text}」")
        for label in ("A_素", "D_ノイズ埋め", "A_素", "D_ノイズ埋め"):
            print(f"  {label}")
            winsound.PlaySound(paths[label], winsound.SND_FILENAME)
        if args.with_zero:
            print("  C_ゼロ埋め（ぶつ切りに聞こえた版。参考）")
            winsound.PlaySound(paths["C_ゼロ埋め"], winsound.SND_FILENAME)
        if args.with_bare:
            print("  B_無音なし（区切りが消えたもの。参考）")
            winsound.PlaySound(paths["B_無音なし"], winsound.SND_FILENAME)


if __name__ == "__main__":
    main()
