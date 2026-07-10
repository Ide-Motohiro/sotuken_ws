"""VOICEVOX から日本語の全モーラ（清音・濁音・半濁音・拗音、計104種）の
短い音声サンプルを一括生成し、`google_stt/mora_bank/` に wav として保存する。

Animalese（どうぶつの森風の疑似言語音）方式の素材集めに相当する処理。
録音済み音声の代わりに VOICEVOX の合成音声をモーラ単位の「録音サンプル」として使う。
実際のクリエイター向けAnimalese制作手順に倣い、各モーラを固定の長さ
（TARGET_DURATION）に正規化して保存する（「あ」「い」「う」…を均一な長さで
録音する、という手順に相当）。

事前条件：VOICEVOX が http://localhost:50021 で起動していること。

使い方：
    python -m google_stt.build_mora_bank                     # デフォルト（ずんだもん ノーマル、speaker=3）
    python -m google_stt.build_mora_bank --speaker 11         # 話者を指定（例：玄野武宏 ノーマル）
"""
import argparse
import io
import os
import wave
import requests
import numpy as np

VOICEVOX_URL = "http://localhost:50021"
DEFAULT_SPEAKER_ID = 3  # ずんだもん ノーマル
BANK_ROOT_DIR = os.path.join(os.path.dirname(__file__), "mora_bank")
TARGET_DURATION = 0.25  # 秒。各モーラをこの長さに正規化する
FLAT_PITCH = 5.8  # VOICEVOXが単独発話ごとに変えてくる抑揚を消し、全モーラ同一のピッチ値に固定する

# 五十音（清音46 + 濁音20 + 半濁音5 + 拗音33 = 104）
MORAS = [
    # 清音
    "あ", "い", "う", "え", "お",
    "か", "き", "く", "け", "こ",
    "さ", "し", "す", "せ", "そ",
    "た", "ち", "つ", "て", "と",
    "な", "に", "ぬ", "ね", "の",
    "は", "ひ", "ふ", "へ", "ほ",
    "ま", "み", "む", "め", "も",
    "や", "ゆ", "よ",
    "ら", "り", "る", "れ", "ろ",
    "わ", "を", "ん",
    # 濁音
    "が", "ぎ", "ぐ", "げ", "ご",
    "ざ", "じ", "ず", "ぜ", "ぞ",
    "だ", "ぢ", "づ", "で", "ど",
    "ば", "び", "ぶ", "べ", "ぼ",
    # 半濁音
    "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
    # 拗音
    "きゃ", "きゅ", "きょ",
    "しゃ", "しゅ", "しょ",
    "ちゃ", "ちゅ", "ちょ",
    "にゃ", "にゅ", "にょ",
    "ひゃ", "ひゅ", "ひょ",
    "みゃ", "みゅ", "みょ",
    "りゃ", "りゅ", "りょ",
    "ぎゃ", "ぎゅ", "ぎょ",
    "じゃ", "じゅ", "じょ",
    "びゃ", "びゅ", "びょ",
    "ぴゃ", "ぴゅ", "ぴょ",
]


def flatten_pitch(query: dict, flat_pitch: float) -> dict:
    """VOICEVOXが単独発話ごとに割り当てる抑揚（ピッチ）を消し、全モーラを同一のピッチ値に揃える"""
    for accent_phrase in query["accent_phrases"]:
        for mora in accent_phrase["moras"]:
            if mora["pitch"] > 0:
                mora["pitch"] = flat_pitch
        pause_mora = accent_phrase.get("pause_mora")
        if pause_mora and pause_mora["pitch"] > 0:
            pause_mora["pitch"] = flat_pitch
    return query


def fetch_mora_wav(mora: str, speaker_id: int) -> bytes:
    query = requests.post(
        f"{VOICEVOX_URL}/audio_query",
        params={"text": mora, "speaker": speaker_id},
    ).json()
    query = flatten_pitch(query, FLAT_PITCH)
    response = requests.post(
        f"{VOICEVOX_URL}/synthesis",
        params={"speaker": speaker_id},
        json=query,
    )
    response.raise_for_status()
    return response.content


def wav_bytes_to_array(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        sampwidth = wf.getsampwidth()
    if sampwidth != 2:
        raise ValueError(f"想定外のサンプル幅です: {sampwidth}")
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, sr


def trim_silence(arr: np.ndarray, threshold: float = 0.02, pad_samples: int = 400) -> np.ndarray:
    """VOICEVOXが前後に付与する無音部分を、閾値ベースで簡易的にトリムする"""
    mag = np.abs(arr)
    above = np.where(mag > threshold)[0]
    if len(above) == 0:
        return arr
    start = max(0, above[0] - pad_samples)
    end = min(len(arr), above[-1] + pad_samples)
    return arr[start:end]


def normalize_duration(arr: np.ndarray, target_duration: float, sr: int) -> np.ndarray:
    """モーラの長さを線形リサンプリングで固定長に正規化する（短い音は伸ばし、長い音は縮める）"""
    target_n = max(1, int(sr * target_duration))
    if len(arr) < 2:
        return np.zeros(target_n, dtype=np.float32)
    idx = np.linspace(0, len(arr) - 1, target_n)
    return np.interp(idx, np.arange(len(arr)), arr).astype(np.float32)


def apply_fade(arr: np.ndarray, fade_samples: int = 200) -> np.ndarray:
    """正規化後の前後の接合部でクリックが出ないよう短いフェードをかける"""
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


def array_to_wav_bytes(arr: np.ndarray, sr: int) -> bytes:
    int16_arr = np.clip(arr * 32768.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_arr.tobytes())
    return buf.getvalue()


def bank_dir_for_speaker(speaker_id: int) -> str:
    if speaker_id == DEFAULT_SPEAKER_ID:
        return BANK_ROOT_DIR
    return f"{BANK_ROOT_DIR}_speaker{speaker_id}"


def build_bank(speaker_id: int) -> None:
    output_dir = bank_dir_for_speaker(speaker_id)
    os.makedirs(output_dir, exist_ok=True)
    print(f"VOICEVOX（speaker={speaker_id}）から{len(MORAS)}モーラを生成します（出力先: {output_dir}）...")

    failed = []
    for i, mora in enumerate(MORAS, start=1):
        out_path = os.path.join(output_dir, f"{mora}.wav")
        try:
            wav_bytes = fetch_mora_wav(mora, speaker_id)
            arr, sr = wav_bytes_to_array(wav_bytes)
            trimmed = trim_silence(arr)
            normalized = normalize_duration(trimmed, TARGET_DURATION, sr)
            faded = apply_fade(normalized)
            with open(out_path, "wb") as f:
                f.write(array_to_wav_bytes(faded, sr))
            print(f"[{i}/{len(MORAS)}] {mora} -> {len(trimmed)/sr*1000:.0f}ms -> {len(faded)/sr*1000:.0f}ms")
        except Exception as e:
            print(f"[{i}/{len(MORAS)}] {mora} 失敗: {e}")
            failed.append(mora)

    print("完了。")
    if failed:
        print(f"失敗したモーラ（{len(failed)}件）: {', '.join(failed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOICEVOXモーラ音声バンクを生成する")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    args = parser.parse_args()
    build_bank(args.speaker)
