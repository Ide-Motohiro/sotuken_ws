"""VOICEVOXの合成レイテンシを実測する計測スクリプト。

疑似言語音を「モーラ素材を並べる（足し算）」方式から「VOICEVOXで普通に合成してから
一様な加工で明瞭度だけを落とす（引き算）」方式へ切り替えるにあたり、合成にかかる
時間がフィラー音声で隠せる範囲かどうかを判断するための計測。

計測するもの：
  - audio_query と synthesis を個別に計測（どちらがボトルネックかの切り分け）
  - テキスト長 4水準 × pitchScale 3水準
  - 各条件 ウォームアップ1回 + 本計測10回（ウォームアップは統計から除外して別途表示）

補足：pitchScale は audio_query のリクエストパラメータではなく、返ってきたクエリJSONの
フィールドである。したがって audio_query の所要時間には原理的に影響せず、効くのは
synthesis 側だけになる（表の audio_query 列は pitchScale 間で同一リクエストの反復測定）。

HTTPの呼び方は infrastructure/voicevox.py の本番実装に合わせ、requests.post を都度
呼ぶ（コネクションを使い回さない）。localhost 前提なので接続確立のオーバーヘッドは
1ms程度に収まるはず、という想定の検証も兼ねている。

事前条件：VOICEVOX が http://localhost:50021 で起動していること。

使い方：
    python -m google_stt.measure_voicevox_latency
    python -m google_stt.measure_voicevox_latency --trials 20        # 試行回数を増やす
    python -m google_stt.measure_voicevox_latency --speaker 11        # 話者を変えて計測
"""
import argparse
import io
import socket
import statistics
import sys
import time
import unicodedata
import urllib.parse
import wave

import requests

from google_stt.infrastructure.voicevox import VoiceVoxTTS

# 話者などの既定値は本番実装（infrastructure/voicevox.py）から借りる。
# VoiceVoxTTS のコンストラクタは値を保持するだけで副作用はない。
_PRODUCTION_DEFAULTS = VoiceVoxTTS()
PRODUCTION_URL = _PRODUCTION_DEFAULTS.url
DEFAULT_SPEAKER_ID = _PRODUCTION_DEFAULTS.speaker_id

# 計測は 127.0.0.1 を既定にする。本番実装が使っている "localhost" はこの環境では
# ::1（IPv6）を先に返し、VOICEVOX が IPv6 で待ち受けていないため、TCP接続のたびに
# 約2秒のフォールバック待ちが入る（実測値は起動時の「接続オーバーヘッドの確認」に出る）。
# その2秒は合成処理とは無関係なので、既定では混入させずに測る。
DEFAULT_VOICEVOX_URL = "http://127.0.0.1:50021"

WARMUP_TRIALS = 1
DEFAULT_TRIALS = 10

# 文字数 5 / 15 / 30 / 60 前後の自然な日本語。実際の文字数は実行時に len() で数えて表示する。
TEXTS = [
    ("最短句", "うん、そうだね"),
    ("短文", "今日はいい天気だから散歩したいね"),
    ("中文", "そのお店なら駅の南口から歩いて五分くらいのところにありますよ"),
    (
        "長文",
        "明日の午後から雨が降るみたいだから、出かけるなら午前中のほうがいいと思うよ。"
        "傘を持っていくなら折りたたみで十分だと思う。",
    ),
]

PITCH_SCALES = [0.0, 0.10, 0.20]


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


def measure_tcp_connect(host: str, port: int, reps: int = 3) -> float:
    """ホスト名でのTCP接続確立にかかる時間[ms]の中央値。名前解決の影響を切り分けるために測る"""
    elapsed = []
    for _ in range(reps):
        started = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=10.0)
        elapsed.append((time.perf_counter() - started) * 1000.0)
        sock.close()
    return statistics.median(elapsed)


def report_connection_overhead(url: str) -> None:
    """合成時間の測定に、名前解決由来の固定オーバーヘッドが混ざっていないか確認して表示する"""
    parsed = urllib.parse.urlparse(url)
    port = parsed.port or 50021

    print("接続オーバーヘッドの確認（TCP接続確立のみ、3回の中央値）")
    measured = {}
    for host in ("127.0.0.1", "localhost"):
        try:
            measured[host] = measure_tcp_connect(host, port)
            print(f"  {host:<12} {measured[host]:7.1f} ms")
        except OSError as e:
            print(f"  {host:<12} 接続失敗: {type(e).__name__}: {e}")

    slow = measured.get("localhost")
    fast = measured.get("127.0.0.1")
    if slow is not None and fast is not None and slow - fast > 100.0:
        print()
        print(f"  ⚠ localhost 経由は 127.0.0.1 より {slow - fast:.0f} ms 遅い。")
        print("    localhost が ::1（IPv6）を先に返し、VOICEVOX が IPv6 で待ち受けていないため、")
        print("    IPv4 へフォールバックするまでの待ちが1リクエストごとに加算されている。")
        print(f"    本番実装（infrastructure/voicevox.py）の接続先は {PRODUCTION_URL} なので、")
        print("    audio_query と synthesis の2リクエスト分がそのまま実運用のレイテンシに乗る。")
    print()


def timed_audio_query(url: str, text: str, speaker_id: int) -> tuple[dict, float]:
    """audio_query を1回投げ、クエリJSONと所要時間[ms]を返す"""
    started = time.perf_counter()
    response = requests.post(
        f"{url}/audio_query",
        params={"text": text, "speaker": speaker_id},
    )
    response.raise_for_status()
    query = response.json()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return query, elapsed_ms


def timed_synthesis(url: str, query: dict, speaker_id: int) -> tuple[bytes, float]:
    """synthesis を1回投げ、wavバイト列と所要時間[ms]を返す"""
    started = time.perf_counter()
    response = requests.post(
        f"{url}/synthesis",
        params={"speaker": speaker_id},
        json=query,
    )
    response.raise_for_status()
    audio = response.content
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return audio, elapsed_ms


def wav_duration_sec(wav_bytes: bytes) -> float:
    """合成された音声そのものの長さ[秒]。合成時間と比べて実時間比を出すために使う"""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def measure_condition(
    url: str, text: str, speaker_id: int, pitch_scale: float, trials: int
) -> dict:
    """1条件（テキスト × pitchScale）を ウォームアップ + trials回 計測する"""
    query_times = []
    synth_times = []
    warmup_query_ms = None
    warmup_synth_ms = None
    audio_sec = 0.0

    for i in range(WARMUP_TRIALS + trials):
        query, query_ms = timed_audio_query(url, text, speaker_id)
        query["pitchScale"] = pitch_scale
        audio, synth_ms = timed_synthesis(url, query, speaker_id)

        if i < WARMUP_TRIALS:
            warmup_query_ms = query_ms
            warmup_synth_ms = synth_ms
            continue

        query_times.append(query_ms)
        synth_times.append(synth_ms)
        audio_sec = wav_duration_sec(audio)

    return {
        "query_times": query_times,
        "synth_times": synth_times,
        "warmup_query_ms": warmup_query_ms,
        "warmup_synth_ms": warmup_synth_ms,
        "audio_sec": audio_sec,
    }


def display_width(s: str) -> int:
    """全角文字を2幅として数える（日本語混じりの表を桁揃えするため）"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, width: int, align: str = "left") -> str:
    space = " " * max(0, width - display_width(s))
    return s + space if align == "left" else space + s


def print_table(headers: list[str], widths: list[int], rows: list[list[str]]) -> None:
    header_line = "  ".join(pad(h, w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * display_width(header_line))
    for row in rows:
        # 先頭列だけ左揃え、数値列は右揃え
        cells = [pad(row[0], widths[0])]
        cells += [pad(c, w, align="right") for c, w in zip(row[1:], widths[1:])]
        print("  ".join(cells))


def stats(values: list[float]) -> tuple[float, float, float]:
    return statistics.median(values), statistics.mean(values), max(values)


def run(url: str, speaker_id: int, trials: int) -> None:
    check_voicevox_running(url)

    total_conditions = len(TEXTS) * len(PITCH_SCALES)
    print(f"VOICEVOX: {url} / speaker={speaker_id}")
    print(
        f"条件数 {total_conditions}（テキスト{len(TEXTS)}水準 × pitchScale{len(PITCH_SCALES)}水準）、"
        f"各条件 ウォームアップ{WARMUP_TRIALS}回 + 本計測{trials}回"
    )
    print()

    report_connection_overhead(url)

    # 全体のウォームアップ（サーバ側の初回コストを最初の条件だけに押し付けないため）
    print("全体ウォームアップ中...")
    warm_query, warm_query_ms = timed_audio_query(url, TEXTS[0][1], speaker_id)
    _, warm_synth_ms = timed_synthesis(url, warm_query, speaker_id)
    print(f"  audio_query {warm_query_ms:.1f} ms / synthesis {warm_synth_ms:.1f} ms（統計からは除外）")
    print()

    results = []
    for label, text in TEXTS:
        for pitch_scale in PITCH_SCALES:
            print(f"計測中: {label}（{len(text)}文字） pitchScale={pitch_scale:.2f} ...", flush=True)
            measured = measure_condition(url, text, speaker_id, pitch_scale, trials)
            results.append(
                {
                    "label": label,
                    "text": text,
                    "chars": len(text),
                    "pitch_scale": pitch_scale,
                    **measured,
                }
            )
    print()

    print_results(results, trials)


def print_results(results: list[dict], trials: int) -> None:
    print("=" * 78)
    print(f"計測結果（各条件 {trials} 試行、ウォームアップ1回は統計から除外）")
    print("=" * 78)
    print()

    print("【1】audio_query 所要時間 [ms]")
    print("     ※ pitchScale はクエリJSONのフィールドでありリクエストには影響しないため、")
    print("        pitchScale 間の差は同一リクエストの測定ばらつきとして読むこと")
    headers = ["テキスト", "文字数", "pitch", "中央値", "平均", "最大", "初回"]
    widths = [10, 6, 6, 8, 8, 8, 8]
    rows = []
    for r in results:
        median, mean, maximum = stats(r["query_times"])
        rows.append(
            [
                r["label"],
                str(r["chars"]),
                f"{r['pitch_scale']:.2f}",
                f"{median:.1f}",
                f"{mean:.1f}",
                f"{maximum:.1f}",
                f"{r['warmup_query_ms']:.1f}",
            ]
        )
    print_table(headers, widths, rows)
    print()

    print("【2】synthesis 所要時間 [ms]")
    rows = []
    for r in results:
        median, mean, maximum = stats(r["synth_times"])
        rows.append(
            [
                r["label"],
                str(r["chars"]),
                f"{r['pitch_scale']:.2f}",
                f"{median:.1f}",
                f"{mean:.1f}",
                f"{maximum:.1f}",
                f"{r['warmup_synth_ms']:.1f}",
            ]
        )
    print_table(headers, widths, rows)
    print()

    print("【3】合計（audio_query + synthesis）と音声長の比較")
    print("     実時間比 = 合計所要時間 ÷ 合成された音声の長さ（1.0未満なら再生に追いつく）")
    headers = ["テキスト", "文字数", "pitch", "合計中央値", "音声長[s]", "実時間比"]
    widths = [10, 6, 6, 12, 10, 10]
    rows = []
    for r in results:
        total_times = [q + s for q, s in zip(r["query_times"], r["synth_times"])]
        total_median = statistics.median(total_times)
        audio_sec = r["audio_sec"]
        rtf = (total_median / 1000.0) / audio_sec if audio_sec > 0 else float("nan")
        rows.append(
            [
                r["label"],
                str(r["chars"]),
                f"{r['pitch_scale']:.2f}",
                f"{total_median:.1f}",
                f"{audio_sec:.2f}",
                f"{rtf:.3f}",
            ]
        )
    print_table(headers, widths, rows)
    print()

    print_headline(results)


def print_headline(results: list[dict]) -> None:
    """フィラーで隠すべき時間の目安＝最短句の synthesis 中央値を目立たせて出す"""
    shortest_label = TEXTS[0][0]
    shortest_text = TEXTS[0][1]
    shortest = [r for r in results if r["label"] == shortest_label]

    print("=" * 78)
    print(f" 最短句「{shortest_text}」（{len(shortest_text)}文字）の synthesis 所要時間の中央値")
    print("=" * 78)
    for r in shortest:
        synth_median = statistics.median(r["synth_times"])
        query_median = statistics.median(r["query_times"])
        print(
            f"   pitchScale {r['pitch_scale']:.2f} :  synthesis {synth_median:7.1f} ms"
            f"   （audio_query 込み合計 {synth_median + query_median:7.1f} ms）"
        )
    print()

    baseline = next(r for r in shortest if r["pitch_scale"] == 0.0)
    baseline_synth = statistics.median(baseline["synth_times"])
    baseline_total = baseline_synth + statistics.median(baseline["query_times"])
    print(f" → フィラーで隠すべき時間の目安： synthesis {baseline_synth:.1f} ms")
    print(f"   （audio_query から数えるなら {baseline_total:.1f} ms、pitchScale=0.00 基準）")
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="VOICEVOXの合成レイテンシを実測する")
    parser.add_argument("--url", type=str, default=DEFAULT_VOICEVOX_URL, help="VOICEVOXのURL")
    parser.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER_ID, help="VOICEVOXの話者ID")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="各条件の本計測回数")
    args = parser.parse_args()

    if args.trials < 2:
        print("--trials は2以上を指定してください（中央値・平均が意味を持たないため）", file=sys.stderr)
        sys.exit(1)

    run(args.url, args.speaker, args.trials)


if __name__ == "__main__":
    main()
