"""VOICEVOX の pitchScale と、既存のリサンプリング方式を聞き比べる比較スクリプト。

疑似言語音を「モーラ素材を並べる（足し算）」方式から「VOICEVOXで普通に合成してから
一様な加工で明瞭度だけを落とす（引き算）」方式へ切り替えるにあたり、
キャラクター性のために声を高くする手段として pitchScale が使えるかを耳で確認する。

確認したい想定：
    pitchScale は合成の前段で F0（声の高さ）を操作するパラメータなので、
    波形を直接リサンプリングする方式と違い、フォルマント（音色・母音の同定性）は
    保たれるはず。この想定が崩れると引き算方式の前提そのものが崩れる。
    ※ VOICEVOX の内部実装を確認したわけではなく、あくまで聴取による検証。

再生する条件：
    1. pitchScale = 0.00 / 0.05 / 0.10 / 0.15 / 0.20（VOICEVOX側でF0を操作）
    2. 比較用：pitchScale=0.00 の音声に既存の resample_speed を x1.3 / x1.5 で適用
       （compare_mora_voice.py からそのまま import して使う。同等品を書き直すと
         「既存方式との比較」にならないため）
    3. 最後に pitchScale=0.20 と resample 版を交互に再生する直接A/B

speedScale・intonationScale は変数を1つに絞るため既定値のまま触らない。

生成した音声は google_stt/pitch_samples/ に wav で保存する（*.wav は .gitignore 対象なので
リポジトリには残らない）。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.compare_pitch_scale
    python -m google_stt.compare_pitch_scale --text "別の文で試す"
    python -m google_stt.compare_pitch_scale --no-play      # 再生せず wav 保存だけ
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import requests
import sounddevice as sd

from google_stt.build_mora_bank import array_to_wav_bytes, wav_bytes_to_array
from google_stt.compare_mora_voice import resample_speed
from google_stt.infrastructure.voicevox import VoiceVoxTTS

# 話者などの既定値は本番実装（infrastructure/voicevox.py）から借りる。
# VoiceVoxTTS のコンストラクタは値を保持するだけで副作用はない。
_PRODUCTION_DEFAULTS = VoiceVoxTTS()
DEFAULT_SPEAKER_ID = _PRODUCTION_DEFAULTS.speaker_id

# 本番実装が使っている "localhost" はこの環境では ::1（IPv6）を先に返し、VOICEVOX が
# IPv6 で待ち受けていないため、TCP接続のたびに約2秒のフォールバック待ちが入る
# （measure_voicevox_latency.py で実測済み）。待たされるだけなので 127.0.0.1 を既定にする。
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "pitch_samples")

# 5母音がひと通り含まれる自然な文（25文字）
SAMPLE_TEXT = "今日はいい天気だから、公園まで歩いて行こうと思う。"

PITCH_SCALES = [0.0, 0.05, 0.10, 0.15, 0.20]
RESAMPLE_FACTORS = [1.3, 1.5]

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


def synthesize(url: str, text: str, speaker_id: int, pitch_scale: float) -> bytes:
    """pitchScale だけを変えて合成する。speedScale・intonationScale は既定値のまま触らない"""
    query_response = requests.post(
        f"{url}/audio_query",
        params={"text": text, "speaker": speaker_id},
    )
    query_response.raise_for_status()
    query = query_response.json()

    query["pitchScale"] = pitch_scale

    synthesis_response = requests.post(
        f"{url}/synthesis",
        params={"speaker": speaker_id},
        json=query,
    )
    synthesis_response.raise_for_status()
    return synthesis_response.content


def save_wav(wav_bytes: bytes, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(wav_bytes)
    return path


def play(arr: np.ndarray, sample_rate: int) -> None:
    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as out_stream:
        out_stream.write(arr)


def announce_and_play(
    header: str, note: str, arr: np.ndarray, sample_rate: int, should_play: bool
) -> None:
    duration = len(arr) / sample_rate
    print(f"{header}（{duration:.2f}秒）")
    if note:
        print(f"    {note}")
    if should_play:
        play(arr, sample_rate)
        time.sleep(PAUSE_SEC)


def run(url: str, text: str, speaker_id: int, should_play: bool) -> None:
    check_voicevox_running(url)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"VOICEVOX: {url} / speaker={speaker_id}")
    print(f"サンプル文: {text}（{len(text)}文字）")
    print(f"保存先: {OUTPUT_DIR}")
    print()
    print("聞き分けるポイント：声の高さが変わっても、母音（あいうえお）が")
    print("同じ母音として聞き取れるか。resample 版で母音が別物に聞こえるなら、")
    print("フォルマントが一緒に動いてしまっている。")
    print()

    # --- 1. pitchScale 条件 ---
    print("=" * 70)
    print("【1】pitchScale を変えて合成（VOICEVOX側でF0を操作）")
    print("=" * 70)

    synthesized = {}
    for pitch_scale in PITCH_SCALES:
        wav_bytes = synthesize(url, text, speaker_id, pitch_scale)
        arr, sample_rate = wav_bytes_to_array(wav_bytes)
        synthesized[pitch_scale] = (arr, sample_rate)

        path = save_wav(wav_bytes, f"pitch_{pitch_scale:.2f}.wav")
        # VOICEVOX のクエリ内の pitch 値は対数周波数なので、pitchScale は F0 の倍率 e^x に効く
        f0_ratio = math.exp(pitch_scale)
        announce_and_play(
            f"pitchScale = {pitch_scale:.2f} を再生します",
            f"F0 は約 x{f0_ratio:.2f} 相当（フォルマントは操作していない） / {os.path.basename(path)}",
            arr,
            sample_rate,
            should_play,
        )
    print()

    # --- 2. 既存の resample 方式（比較対象） ---
    print("=" * 70)
    print("【2】比較：既存の resample_speed 方式（波形を直接リサンプリング）")
    print("     pitchScale=0.00 の音声に適用。速度・F0・フォルマントが同時に動く")
    print("=" * 70)

    base_arr, base_sample_rate = synthesized[0.0]
    for factor in RESAMPLE_FACTORS:
        resampled = resample_speed(base_arr, factor)
        path = save_wav(
            array_to_wav_bytes(resampled, base_sample_rate), f"resample_{factor}x.wav"
        )
        announce_and_play(
            f"resample x{factor} を再生します",
            f"F0 も フォルマントも x{factor:.2f}、再生時間は 1/{factor:.2f} / {os.path.basename(path)}",
            resampled,
            base_sample_rate,
            should_play,
        )
    print()

    if not should_play:
        print("（--no-play のため再生はしていない。保存した wav を聞き比べること）")
        return

    # --- 3. 直接A/B ---
    print("=" * 70)
    print("【3】直接A/B：同じくらい音が高くなる操作どうしを交互に再生")
    print(f"     pitchScale 0.20（F0 約 x{math.exp(0.20):.2f}）↔ resample x1.3（F0 x1.30）")
    print("=" * 70)

    top_pitch_arr, top_pitch_sample_rate = synthesized[0.20]
    resampled_13 = resample_speed(base_arr, 1.3)
    for round_index in (1, 2):
        announce_and_play(
            f"[{round_index}周目 A] pitchScale = 0.20",
            "",
            top_pitch_arr,
            top_pitch_sample_rate,
            should_play,
        )
        announce_and_play(
            f"[{round_index}周目 B] resample x1.3",
            "",
            resampled_13,
            base_sample_rate,
            should_play,
        )
    print()
    print("完了。保存済みの wav で何度でも聞き直せる:")
    print(f"  {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VOICEVOXのpitchScaleとresample方式を聞き比べる")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument("--text", type=str, default=SAMPLE_TEXT, help="読み上げるサンプル文")
    parser.add_argument("--no-play", action="store_true", help="再生せず wav の保存だけを行う")
    args = parser.parse_args()

    run(args.url, args.text, args.speaker, should_play=not args.no_play)


if __name__ == "__main__":
    main()
