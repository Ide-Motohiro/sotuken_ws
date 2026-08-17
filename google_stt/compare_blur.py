"""疑似言語音のための「明瞭度を落とす加工」を聞き比べる比較スクリプト。

方針：モーラ素材を並べる（足し算）のではなく、VOICEVOXで普通に合成した音声に対して
一様な加工をかけて明瞭度だけを落とす（引き算）。写真にモザイクをかけて粗さを調整する
イメージで、「字幕なしでもふんわり伝わる」水準のつまみを探す。

比較する3つの加工（それぞれ強度3段階）：
  (a) スペクトルのぼかし   STFTの振幅を周波数方向に移動平均し、フォルマントの山をなだらかにする
  (b) ノイズボコーダ       N帯域に分けて各帯域の振幅包絡だけを取り出し、その包絡でノイズを変調する
  (c) 子音の立ち上がり抑制 振幅の急激な立ち上がり（トランジェント）を検出して減衰させる

いずれも入力テキストの内容では一切分岐しない、全入力に同一の手続きを適用する静的な処理。
ノイズボコーダのノイズ源も一様分布から生成し、既定では固定シードで再現性を持たせている。

サンプルレートについて：VOICEVOX の出力は 24000Hz。再生・保存はすべて合成結果の wav から
読み取った実際のサンプルレートを使う（PseudoVoiceTTS の 44100Hz とは無関係）。

依存ライブラリは numpy のみ（scipy / librosa は使わない）。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.compare_blur
    python -m google_stt.compare_blur --no-play    # 再生せず wav 保存だけ
"""
import argparse
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "blur_samples")

SAMPLE_TEXTS = [
    ("short", "うんうん、そうだね。"),
    ("long", "そのお店なら駅の南口から歩いて五分くらいのところにありますよ"),
]

FFT_SIZE = 1024
HOP_SIZE = 256

# (a) スペクトルのぼかし：振幅を周波数方向に移動平均する幅[ビン]。24000Hz/1024点なら1ビン≒23Hz
SPECTRAL_BLUR_LEVELS = [("weak", 5), ("mid", 15), ("strong", 41)]

# (b) ノイズボコーダ：帯域数。少ないほどスペクトル情報が粗くなる
VOCODER_BAND_COUNTS = [8, 4, 2]
VOCODER_FMIN = 100.0

# (c) 子音の立ち上がり抑制：音量が上がってよい速さの上限[dB/ms]。小さいほどアタックが鈍る。
# 自然な発話の子音の立ち上がりは概ね 2dB/ms 前後なので、それより遅い値を並べている。
TRANSIENT_LEVELS = [("weak", 1.2), ("mid", 0.6), ("strong", 0.25)]
TRANSIENT_HOP = 64        # 包絡の時間分解能[サンプル]。24000Hzなら約2.7ms
TRANSIENT_SMOOTH_MS = 4.0  # ゲイン変化を滑らかにしてクリックを防ぐ

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


def synthesize(url: str, text: str, speaker_id: int) -> bytes:
    """VOICEVOXの既定パラメータのまま合成する（pitchScale等は触らない）"""
    query_response = requests.post(
        f"{url}/audio_query",
        params={"text": text, "speaker": speaker_id},
    )
    query_response.raise_for_status()

    synthesis_response = requests.post(
        f"{url}/synthesis",
        params={"speaker": speaker_id},
        json=query_response.json(),
    )
    synthesis_response.raise_for_status()
    return synthesis_response.content


# --- STFT / ISTFT（numpyのみ。窓の二乗和で割る重み付きoverlap-add） ---

def _hann(size: int) -> np.ndarray:
    """周期窓。hop = size/4 の重み付きOLAで正確に再構成できる"""
    return np.hanning(size + 1)[:size]


def stft(x: np.ndarray, fft_size: int = FFT_SIZE, hop: int = HOP_SIZE) -> tuple[np.ndarray, int]:
    """(フレーム数, 周波数ビン) の複素スペクトログラムと、元の長さを返す"""
    window = _hann(fft_size)
    padded = np.concatenate([np.zeros(fft_size), x, np.zeros(2 * fft_size)])
    n_frames = 1 + (len(padded) - fft_size) // hop
    frames = np.stack([padded[i * hop:i * hop + fft_size] * window for i in range(n_frames)])
    return np.fft.rfft(frames, axis=1), len(x)


def istft(spectrogram: np.ndarray, original_length: int,
          fft_size: int = FFT_SIZE, hop: int = HOP_SIZE) -> np.ndarray:
    window = _hann(fft_size)
    frames = np.fft.irfft(spectrogram, n=fft_size, axis=1)
    total = (len(frames) - 1) * hop + fft_size
    out = np.zeros(total)
    weight = np.zeros(total)
    for i, frame in enumerate(frames):
        out[i * hop:i * hop + fft_size] += frame * window
        weight[i * hop:i * hop + fft_size] += window ** 2
    out /= np.maximum(weight, 1e-8)
    return out[fft_size:fft_size + original_length]


def verify_stft_roundtrip(sample_rate: int, rng: np.random.Generator) -> float:
    """STFT→ISTFT が元波形を復元できることを確認する。
    加工なしとの差が加工由来であることを担保するための自己チェック。"""
    probe = rng.uniform(-0.5, 0.5, sample_rate // 2)
    spectrogram, length = stft(probe)
    return float(np.max(np.abs(istft(spectrogram, length) - probe)))


# --- 音量そろえ ---

def match_rms(processed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """加工音の音量を基準音に合わせる。聴き比べで音量差が交絡しないようにするため"""
    processed_rms = float(np.sqrt(np.mean(processed ** 2)))
    reference_rms = float(np.sqrt(np.mean(reference ** 2)))
    if processed_rms < 1e-9:
        return processed
    scaled = processed * (reference_rms / processed_rms)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.99:  # クリップ防止
        scaled = scaled * (0.99 / peak)
    return scaled


# --- (a) スペクトルのぼかし ---

def spectral_blur(x: np.ndarray, width_bins: int) -> np.ndarray:
    """STFTの振幅スペクトルを周波数方向に移動平均し、位相は元のまま戻す。
    フォルマントの山がなだらかになり、母音の区別が曖昧になる。"""
    spectrogram, length = stft(x)
    magnitude = np.abs(spectrogram)
    phase = np.exp(1j * np.angle(spectrogram))

    kernel = np.ones(width_bins) / width_bins
    # 端を反射で延長してから畳み込み、両端のビンが不当に減衰しないようにする
    padding = width_bins // 2
    extended = np.pad(magnitude, ((0, 0), (padding, padding)), mode="reflect")
    smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, extended)

    return match_rms(istft(smoothed * phase, length), x)


# --- (b) ノイズボコーダ ---

def _band_edges(n_bands: int, sample_rate: int, n_bins: int) -> list[tuple[int, int]]:
    """対数間隔で帯域の境界ビンを決める（聴覚の周波数分解に合わせるため対数間隔）"""
    nyquist = sample_rate / 2
    edges_hz = np.exp(np.linspace(np.log(VOCODER_FMIN), np.log(nyquist), n_bands + 1))
    edges_bin = np.clip((edges_hz / nyquist * (n_bins - 1)).astype(int), 0, n_bins - 1)
    bands = []
    for i in range(n_bands):
        low, high = edges_bin[i], max(edges_bin[i + 1], edges_bin[i] + 1)
        bands.append((low, min(high, n_bins)))
    return bands


def noise_vocoder(x: np.ndarray, n_bands: int, sample_rate: int,
                  rng: np.random.Generator) -> np.ndarray:
    """N帯域に分割し、各帯域の振幅包絡でノイズを変調して足し合わせる。
    帯域数が少ないほどスペクトルの手がかりが減り、リズムだけが残る。"""
    spectrogram, length = stft(x)
    magnitude = np.abs(spectrogram)

    # ノイズ源は一様分布から生成する
    noise = rng.uniform(-1.0, 1.0, len(x))
    noise_spectrogram, _ = stft(noise)

    out = np.zeros_like(spectrogram)
    for low, high in _band_edges(n_bands, sample_rate, magnitude.shape[1]):
        # 帯域ごとのフレーム単位の振幅（＝包絡）を、ノイズ側の同じ帯域の振幅に転写する
        signal_band = np.sqrt(np.mean(magnitude[:, low:high] ** 2, axis=1))
        noise_band = np.sqrt(np.mean(np.abs(noise_spectrogram[:, low:high]) ** 2, axis=1))
        gain = signal_band / np.maximum(noise_band, 1e-9)
        out[:, low:high] = noise_spectrogram[:, low:high] * gain[:, None]

    return match_rms(istft(out, length), x)


# --- (c) 子音の立ち上がり抑制 ---

def transient_suppression(x: np.ndarray, max_rise_db_per_ms: float, sample_rate: int) -> np.ndarray:
    """振幅包絡が上昇してよい速さに上限をかけ、アタックを鈍らせる。

    立ち上がり位置のゲインを下げる方式だと「へこみ→復帰」が新たな急上昇を作ってしまい、
    子音が柔らぐどころか途切れて聞こえる。ここでは包絡そのものの上昇レートを制限するので、
    立ち上がりは必ずなだらかになる。減衰しか行わない（gain <= 1）。

    レート上限は固定値で、入力内容による分岐は持たない。"""
    hop = TRANSIENT_HOP
    n_frames = max(1, len(x) // hop)

    frame_rms = np.array([
        np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n_frames)
    ])
    frame_db = 20.0 * np.log10(frame_rms + 1e-12)

    # 1フレームあたりに許す上昇量へ換算し、包絡を上向きにだけ鈍らせる
    max_rise_per_frame = max_rise_db_per_ms * (hop / sample_rate) * 1000.0
    limited_db = np.empty_like(frame_db)
    limited_db[0] = frame_db[0]
    for i in range(1, n_frames):
        limited_db[i] = min(frame_db[i], limited_db[i - 1] + max_rise_per_frame)

    frame_gain = 10.0 ** ((limited_db - frame_db) / 20.0)

    # ゲインをサンプル単位に引き伸ばしてから平滑化し、切り替わりのクリックを防ぐ
    gain = np.interp(
        np.arange(len(x)),
        np.arange(n_frames) * hop + hop / 2,
        frame_gain,
        left=frame_gain[0],
        right=frame_gain[-1],
    )
    smooth_len = max(1, int(sample_rate * TRANSIENT_SMOOTH_MS / 1000.0))
    smoothing_kernel = np.ones(smooth_len) / smooth_len
    gain = np.convolve(gain, smoothing_kernel, mode="same")

    return match_rms(x * gain, x)


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


def announce_and_play(label: str, note: str, arr: np.ndarray,
                      sample_rate: int, should_play: bool) -> None:
    print(f"  {label}（{len(arr) / sample_rate:.2f}秒）")
    if note:
        print(f"      {note}")
    if should_play:
        play(arr, sample_rate)
        time.sleep(PAUSE_SEC)


def build_variants(base: np.ndarray, sample_rate: int, rng: np.random.Generator) -> list[dict]:
    """再生順に並べた条件リストを作る。各グループの頭には基準音（加工なし）を挟む"""
    variants = [
        {"group": "基準", "label": "加工なし（基準音）", "note": "", "file": "original", "audio": base},
    ]

    for name, width in SPECTRAL_BLUR_LEVELS:
        hz = width * (sample_rate / 2) / (FFT_SIZE // 2)
        variants.append({
            "group": "(a) スペクトルのぼかし",
            "label": f"スペクトルのぼかし [{name}]",
            "note": f"平滑化幅 {width}ビン（約{hz:.0f}Hz）",
            "file": f"spectral_blur_{name}",
            "audio": spectral_blur(base, width),
        })

    variants.append({"group": "基準", "label": "加工なし（基準音・耳のリセット）",
                     "note": "", "file": None, "audio": base})

    for n_bands in VOCODER_BAND_COUNTS:
        variants.append({
            "group": "(b) ノイズボコーダ",
            "label": f"ノイズボコーダ [N={n_bands}]",
            "note": f"{n_bands}帯域の振幅包絡のみを保持（{VOCODER_FMIN:.0f}Hz以上を対数間隔で分割）",
            "file": f"vocoder_n{n_bands}",
            "audio": noise_vocoder(base, n_bands, sample_rate, rng),
        })

    variants.append({"group": "基準", "label": "加工なし（基準音・耳のリセット）",
                     "note": "", "file": None, "audio": base})

    for name, max_rise in TRANSIENT_LEVELS:
        variants.append({
            "group": "(c) 子音の立ち上がり抑制",
            "label": f"立ち上がり抑制 [{name}]",
            "note": f"音量の上昇を最大 {max_rise:.2f}dB/ms に制限（小さいほどアタックが鈍る）",
            "file": f"transient_{name}",
            "audio": transient_suppression(base, max_rise, sample_rate),
        })

    return variants


def run(url: str, speaker_id: int, should_play: bool, seed: int) -> None:
    check_voicevox_running(url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rng = np.random.default_rng(seed)
    print(f"VOICEVOX: {url} / speaker={speaker_id}（パラメータは既定値のまま）")
    print(f"保存先: {OUTPUT_DIR}")

    roundtrip_error = verify_stft_roundtrip(24000, rng)
    print(f"STFT往復の最大誤差: {roundtrip_error:.2e}（加工なしとの差が加工由来であることの確認）")
    if roundtrip_error > 1e-6:
        print("STFTの再構成が正しくない。加工結果を信用できないので中断する。", file=sys.stderr)
        sys.exit(1)
    print()

    for text_label, text in SAMPLE_TEXTS:
        wav_bytes = synthesize(url, text, speaker_id)
        base, sample_rate = wav_bytes_to_array(wav_bytes)

        print("=" * 72)
        print(f"サンプル文［{text_label}］: {text}（{len(text)}文字 / {sample_rate}Hz）")
        print("=" * 72)

        variants = build_variants(base, sample_rate, rng)

        current_group = None
        for variant in variants:
            if variant["group"] != current_group:
                current_group = variant["group"]
                print(f"\n--- {current_group} ---")
            if variant["file"]:
                save_wav(variant["audio"], sample_rate, f"{text_label}_{variant['file']}.wav")
            announce_and_play(variant["label"], variant["note"],
                              variant["audio"], sample_rate, should_play)
        print()

    print("完了。保存済みの wav で何度でも聞き直せる:")
    print(f"  {OUTPUT_DIR}")
    if not should_play:
        print("（--no-play のため再生はしていない）")


def main() -> None:
    parser = argparse.ArgumentParser(description="疑似言語音の明瞭度を落とす加工を聞き比べる")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument("--seed", type=int, default=0, help="ノイズ源のシード（再現性のため固定）")
    parser.add_argument("--no-play", action="store_true", help="再生せず wav の保存だけを行う")
    args = parser.parse_args()

    run(args.url, args.speaker, should_play=not args.no_play, seed=args.seed)


if __name__ == "__main__":
    main()
