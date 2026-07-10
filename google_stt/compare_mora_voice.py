"""VOICEVOXで事前生成した固定長モーラ音声バンク（`google_stt/mora_bank/`）を使い、
実際のAnimalese（どうぶつの森風疑似言語音）制作手順を再現して聞き比べる一時スクリプト。

再現している手順：
  1. 各モーラを均一な長さに正規化した音声素材を並べ、隙間なく連結する（build_mora_bank.py側で対応済み）
  2. 全体を一律の倍率で高速化する（ピッチも連動して上がる）
  3. モーラ1つ1つにランダムなピッチのばらつきを追加でかける

事前条件：`python -m google_stt.build_mora_bank [--speaker ID]` で固定長モーラバンクを生成済みであること。

使い方：
    python -m google_stt.compare_mora_voice                # デフォルト（ずんだもん、mora_bank/）
    python -m google_stt.compare_mora_voice --speaker 11    # 別話者のバンクを比較（例：玄野武宏）
"""
import argparse
import os
import time
import wave
import numpy as np
import sounddevice as sd
import pykakasi

from google_stt.build_mora_bank import DEFAULT_SPEAKER_ID, bank_dir_for_speaker

SAMPLE_TEXT = "こんにちは、今日はいい天気ですね。何かお手伝いできることはありますか？"

# 比較する基本速度倍率（全体に一律でかける）
BASE_SPEEDS = [2.5, 3.0]
# 比較するモーラごとのランダムピッチのばらつき幅（0.0=ばらつきなし、0.3なら±30%の範囲でランダム）
PITCH_JITTER_LEVELS = [0.0, 0.15, 0.3]

SMALL_YOON = set("ゃゅょ")
SOKUON = "っ"
CHOON = "ー"
PAUSE_CHARS = set(" 　\n\r\t、。！？「」『』,.!?")

_kakasi = pykakasi.kakasi()


def to_hiragana_reading(text: str) -> str:
    return "".join(item["hira"] for item in _kakasi.convert(text))


def split_moras(hiragana: str) -> list[str]:
    """ひらがな文字列をモーラ単位に分割する（拗音「きゃ」等は1モーラとして結合）"""
    moras = []
    i = 0
    n = len(hiragana)
    while i < n:
        c = hiragana[i]
        if i + 1 < n and hiragana[i + 1] in SMALL_YOON and c not in PAUSE_CHARS:
            moras.append(c + hiragana[i + 1])
            i += 2
        else:
            moras.append(c)
            i += 1
    return moras


def load_mora_bank(bank_dir: str) -> dict[str, tuple[np.ndarray, int]]:
    if not os.path.isdir(bank_dir):
        raise FileNotFoundError(
            f"モーラバンクが見つかりません: {bank_dir}\n"
            "先に `python -m google_stt.build_mora_bank` を実行してください。"
        )
    bank = {}
    for fname in os.listdir(bank_dir):
        if not fname.endswith(".wav"):
            continue
        mora = fname[:-4]
        path = os.path.join(bank_dir, fname)
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        bank[mora] = (arr, sr)
    return bank


def resample_speed(arr: np.ndarray, speed_factor: float) -> np.ndarray:
    """波形をそのままリサンプリングして再生速度とピッチを同時に変える（Animalese方式）"""
    if speed_factor == 1.0 or len(arr) < 2:
        return arr
    n = len(arr)
    new_n = max(1, int(n / speed_factor))
    idx = np.linspace(0, n - 1, new_n)
    return np.interp(idx, np.arange(n), arr).astype(np.float32)


def apply_fade(arr: np.ndarray, fade_samples: int = 80) -> np.ndarray:
    """モーラ間の接続でクリックノイズが出ないよう、短いフェードイン/アウトを付与する"""
    n = len(arr)
    if n < 2:
        return arr
    fs = min(fade_samples, n // 2)
    if fs <= 0:
        return arr
    arr = arr.copy()
    arr[:fs] *= np.linspace(0, 1, fs)
    arr[-fs:] *= np.linspace(1, 0, fs)
    return arr


def synthesize(
    hiragana: str,
    bank: dict,
    base_speed: float,
    sr: int,
    pitch_jitter: float = 0.0,
    pause_sec: float = 0.05,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """全体を base_speed で一律高速化しつつ、モーラごとに ±pitch_jitter の範囲で
    追加のランダムピッチ変化をかけて連結する。
    """
    moras = split_moras(hiragana)
    pause = np.zeros(int(sr * pause_sec), dtype=np.float32)
    long_pause = np.zeros(int(sr * pause_sec * 3), dtype=np.float32)

    segments = []
    unknown = set()
    prev_mora_signal = None

    for m in moras:
        if m in PAUSE_CHARS:
            segments.append(long_pause)
            prev_mora_signal = None
            continue
        if m == SOKUON:
            segments.append(pause)  # 促音：短い無音の間
            continue
        if m == CHOON:
            # 長音：直前モーラの音を繰り返して伸ばす簡易対応
            if prev_mora_signal is not None:
                segments.append(prev_mora_signal)
            continue
        if m not in bank:
            unknown.add(m)
            segments.append(pause)
            continue

        raw_arr, _ = bank[m]
        factor = base_speed
        if pitch_jitter > 0 and rng is not None:
            factor = base_speed * rng.uniform(1 - pitch_jitter, 1 + pitch_jitter)
        shifted = resample_speed(raw_arr, factor)
        faded = apply_fade(shifted)
        segments.append(faded)
        prev_mora_signal = faded

    if unknown:
        print(f"  (バンクに存在しないモーラをスキップしました: {', '.join(sorted(unknown))})")

    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


def play(arr: np.ndarray, sr: int) -> None:
    with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out_stream:
        out_stream.write(arr)


def main():
    parser = argparse.ArgumentParser(description="モーラバンク疑似言語音の聞き比べ")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID（バンク生成時と合わせる）")
    args = parser.parse_args()

    bank_dir = bank_dir_for_speaker(args.speaker)
    bank = load_mora_bank(bank_dir)
    sr = next(iter(bank.values()))[1]
    print(f"モーラバンク読み込み完了（{bank_dir}、{len(bank)}種類、サンプルレート{sr}Hz）")

    hiragana = to_hiragana_reading(SAMPLE_TEXT)
    print(f"読み: {hiragana}\n")

    rng = np.random.default_rng()

    for base_speed in BASE_SPEEDS:
        for jitter in PITCH_JITTER_LEVELS:
            print(f"【基本速度 x{base_speed} / ピッチのばらつき ±{jitter*100:.0f}%】を再生します...")
            arr = synthesize(hiragana, bank, base_speed, sr, pitch_jitter=jitter, rng=rng)
            play(arr, sr)
            time.sleep(0.8)

    print("完了。")


if __name__ == "__main__":
    main()
