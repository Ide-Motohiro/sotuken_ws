"""自分の声を1モーラずつ録音して、モーラ音声バンクを作るツール。

VOICEVOXの機械音声(build_mora_bank.py)の代わりに、人間の肉声を素材にする。
Animalese(どうぶつの森風疑似言語音)の元ネタも人の声を録音して速くする方式なので、
肉声のほうが素材として本来的で、ずんだもんの声質(高すぎる・特徴的すぎる)も回避できる。

処理は build_mora_bank.py と共通:各モーラを無音トリム→固定長(TARGET_DURATION)へ
正規化→フェード、で保存する。生成後は compare_mora_voice.py の合成ロジックがそのまま使える。

使い方(自分のターミナルで対話的に実行する):
    python -m google_stt.record_mora_bank                 # mora_bank_myvoice/ に全104モーラを録音
    python -m google_stt.record_mora_bank --name taro      # 出力先を mora_bank_taro/ にする
    python -m google_stt.record_mora_bank --only あ,い,う   # 指定モーラだけ録り直す
    python -m google_stt.record_mora_bank --seion          # 清音46モーラだけ録る

操作:
    各モーラで Enter=録音開始 / s=スキップ / q=中断(ここまでを保存して終了)
    録音後   Enter=採用 / r=録り直し / s=スキップ
"""
import argparse
import os
import sys
import numpy as np
import sounddevice as sd

from google_stt.build_mora_bank import (
    MORAS,
    TARGET_DURATION,
    trim_silence,
    fit_duration_crop,
    apply_fade,
    array_to_wav_bytes,
)

_QUIT = object()  # 「中断」を表すセンチネル（numpy配列との == 比較を避けるため is で判定する）

REC_SAMPLE_RATE = 24000  # VOICEVOXバンクと揃える
REC_WINDOW_SEC = 1.2  # 1モーラの録音窓(この間に一度だけ発声する)
SILENCE_MAX_THRESHOLD = 0.03  # 録音全体のピークがこれ未満なら「ほぼ無音」と判定
BANK_ROOT = os.path.dirname(__file__)

# 清音46モーラ(MORASの先頭46件)
SEION_COUNT = 46


def bank_dir_for_name(name: str) -> str:
    return os.path.join(BANK_ROOT, f"mora_bank_{name}")


def record_window(duration: float, sr: int) -> np.ndarray:
    print("  🔴 録音中... 今すぐ発声してください")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0].copy()


def playback(arr: np.ndarray, sr: int) -> None:
    sd.play(arr, sr)
    sd.wait()


def process_recording(raw: np.ndarray, sr: int) -> np.ndarray:
    """無音トリム→固定長へ切り出し（ピッチ保持）→フェード。
    肉声は録るたびに長さがバラつくため、引き伸ばしではなく切り出しで長さを揃え、
    モーラごとのピッチのヨレを防ぐ。"""
    trimmed = trim_silence(raw)
    fitted = fit_duration_crop(trimmed, TARGET_DURATION, sr)
    return apply_fade(fitted)


def prompt(msg: str) -> str:
    try:
        return input(msg).strip().lower()
    except EOFError:
        # 非対話環境で実行された場合の保険
        print("\n対話的な入力が必要です。自分のターミナルで実行してください。", file=sys.stderr)
        sys.exit(1)


def record_one(mora: str, index: int, total: int, sr: int):
    """1モーラを録音して確定した波形を返す。スキップされたら None、中断なら _QUIT。"""
    while True:
        print(f"\n[{index}/{total}] 「{mora}」")
        cmd = prompt("  Enter=録音 / s=スキップ / q=中断保存: ")
        if cmd == "q":
            return _QUIT
        if cmd == "s":
            print("  → スキップ")
            return None

        raw = record_window(REC_WINDOW_SEC, sr)
        peak = float(np.max(np.abs(raw))) if len(raw) else 0.0
        if peak < SILENCE_MAX_THRESHOLD:
            print(f"  ⚠ ほぼ無音でした(ピーク {peak:.3f})。録り直します。")
            continue

        processed = process_recording(raw, sr)
        print(f"  ▶ 再生して確認(ピーク {peak:.3f})...")
        playback(processed, sr)

        cmd2 = prompt("  Enter=採用 / r=録り直し / s=スキップ: ")
        if cmd2 == "r":
            continue
        if cmd2 == "s":
            print("  → スキップ")
            return None
        return processed


def main():
    parser = argparse.ArgumentParser(description="自分の声でモーラ音声バンクを録音する")
    parser.add_argument("--name", type=str, default="myvoice", help="バンク名(出力先 mora_bank_<name>/)")
    parser.add_argument("--only", type=str, default=None, help="録音するモーラをカンマ区切りで指定(例: あ,い,う)")
    parser.add_argument("--seion", action="store_true", help="清音46モーラだけを対象にする")
    parser.add_argument("--overwrite", action="store_true", help="既に録音済みのモーラも録り直す")
    args = parser.parse_args()

    if args.only:
        targets = [m.strip() for m in args.only.split(",") if m.strip()]
        unknown = [m for m in targets if m not in MORAS]
        if unknown:
            print(f"未知のモーラが指定されました: {', '.join(unknown)}", file=sys.stderr)
            sys.exit(1)
    elif args.seion:
        targets = MORAS[:SEION_COUNT]
    else:
        targets = MORAS

    output_dir = bank_dir_for_name(args.name)
    os.makedirs(output_dir, exist_ok=True)

    # 既存録音をスキップする場合の対象絞り込み
    if not args.overwrite and not args.only:
        pending = [m for m in targets if not os.path.exists(os.path.join(output_dir, f"{m}.wav"))]
        skipped = len(targets) - len(pending)
        if skipped:
            print(f"既に録音済みの {skipped} モーラをスキップします(録り直すには --overwrite)。")
        targets = pending

    if not targets:
        print("録音対象がありません。全モーラ録音済みです。")
        return

    print("=" * 50)
    print(f"出力先: {output_dir}")
    print(f"録音対象: {len(targets)} モーラ / サンプルレート {REC_SAMPLE_RATE}Hz")
    print("各モーラは平坦なトーンで短く発声してください(抑揚をつけない)。")
    print("=" * 50)

    recorded = 0
    for i, mora in enumerate(targets, start=1):
        result = record_one(mora, i, len(targets), REC_SAMPLE_RATE)
        if result is _QUIT:
            print("\n中断しました。ここまでの録音を保存済みです。")
            break
        if result is None:
            continue
        out_path = os.path.join(output_dir, f"{mora}.wav")
        with open(out_path, "wb") as f:
            f.write(array_to_wav_bytes(result, REC_SAMPLE_RATE))
        recorded += 1
        print(f"  ✓ 保存: {mora}.wav")

    print(f"\n完了。{recorded} モーラを録音しました。")
    total_in_bank = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    print(f"バンク内の総モーラ数: {total_in_bank}/{len(MORAS)}")
    print(f"聞き比べ: python -m google_stt.compare_mora_voice --bank {args.name}")


if __name__ == "__main__":
    main()
