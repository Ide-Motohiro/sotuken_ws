"""フィラー音を実際に鳴らして確かめるための再生ツール。

本番と同じ `VoiceVoxFillerPlayer` をそのまま使うので、ここで聞こえる音・間の取り方は
対話中に鳴るものと同一。パラメータをここで決めて、そのまま
`main_phoneme_swap.py --filler-interval` に渡せる。

2つのモードがある。

  既定（--simulate なし）：
      Ctrl+C で止めるまでフィラーを鳴らし続ける。「えーと」→ 間 →「うーん」→ 間 → …
      間の取り方（--interval）を耳で決めるためのモード。

  --simulate SEC：
      実際の1ターンを再現する。フィラーを鳴らし始め、SEC 秒後に「応答ができた」ことにして
      止め、続けて応答の音声を鳴らす。**フィラーから応答へのつなぎ目**と、
      止めるのにかかった遅延を確認できる。
      無音区間の実測は約2.1〜2.5秒なので、--simulate 2.4 あたりが実際に近い。

事前条件：VOICEVOX が起動していること。

使い方：
    python -m google_stt.play_filler                      # 鳴らし続ける（既定の間隔）
    python -m google_stt.play_filler --interval 2.5       # 間隔を変えて聞く
    python -m google_stt.play_filler --delay 0.6          # 立ち上がりを遅らせて聞く
    python -m google_stt.play_filler --simulate 2.4       # 1ターンを再現する
    python -m google_stt.play_filler --simulate 2.0 --compare-interjection  # 応答冒頭の感動詞あり／なし
    python -m google_stt.play_filler --phrases えーと うーん あのー
"""
import argparse
import io
import sys
import time
import wave

from google_stt.domain.models import ConsonantSubstitutionTable
from google_stt.infrastructure.filler import DEFAULT_FILLER_PHRASES, VoiceVoxFillerPlayer
from google_stt.infrastructure.phoneme_swap import PhonemeSwapTTS

#: --simulate で鳴らす応答。冒頭の感動詞（ルール3）を含む形と含まない形を並べてある。
#: フィラーも応答冒頭もどちらも感動詞なので「言い淀み→反応」という人と逆の順番になる。
#: これが違和感の原因かを聞き分けるために --compare-interjection で並べて鳴らす。
SAMPLE_REPLY = "うん、きもち、いいね"
SAMPLE_REPLY_WITHOUT_INTERJECTION = "きもち、いいね"


class AnnouncingFillerPlayer(VoiceVoxFillerPlayer):
    """どのフィラーを鳴らしているか表示するだけの派生。鳴らし方は親のまま"""

    def _next_path(self) -> str:
        # 選択は親に任せ、選ばれた位置（_last_index）を見てから表示する
        path = super()._next_path()
        elapsed = time.perf_counter() - self._started_at
        print(f"  [{elapsed:5.2f}s] 「{self.phrases[self._last_index]}」")
        return path

    def start(self) -> None:
        self._started_at = time.perf_counter()
        super().start()


def wav_duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.getnframes() / w.getframerate()


def run(phrases, interval, delay, simulate, reply, compare, swap_ratio, seed):
    tts = PhonemeSwapTTS(
        substitution_table=ConsonantSubstitutionTable.articulatory(swap_ratio=swap_ratio))

    print(f"フィラーを事前合成中（{len(phrases)}パターン）...")
    started = time.perf_counter()
    try:
        player = AnnouncingFillerPlayer(
            tts, phrases=phrases, repeat_interval_sec=interval,
            initial_delay_sec=delay, seed=seed)
    except Exception as e:
        print(f"VOICEVOXに接続できないか、合成に失敗しました: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(1)
    print(f"完了（{time.perf_counter() - started:.2f}秒）")

    for phrase, path in zip(phrases, player._paths):
        with open(path, "rb") as f:
            print(f"  「{phrase}」{wav_duration(f.read()):.3f}秒")
    print(f"立ち上がり: {delay}秒（認識確定から1つ目を鳴らし始めるまで）")
    print(f"間隔: {interval}秒（1つ鳴らし終えてから次を鳴らすまで）\n")

    try:
        if simulate is None:
            print("Ctrl+C で終了。\n")
            player.start()
            while True:
                time.sleep(0.2)
        else:
            replies = [reply]
            if compare:
                replies = [SAMPLE_REPLY, SAMPLE_REPLY_WITHOUT_INTERJECTION]
                print("感動詞あり／なしを続けて鳴らす。フィラーとの並びを聞き比べる。")
            for text in replies:
                print(f"1ターンの再現：{simulate}秒後に応答ができたことにして切り替える")
                wav_bytes, _, _ = tts.synthesize(text)  # 先に用意して合成待ちを挟まない
                player.start()
                time.sleep(simulate)
                player.stop()
                print(f"  止めるのにかかった遅延: {player.last_stop_delay_sec * 1000:.0f}ms")
                if player.last_stop_delay_sec > 0.05:
                    print("  → フィラーが鳴っている最中に応答ができたため、鳴り終わるまで待った。")
                    print("     この分だけ応答が遅れる。間隔を伸ばすと起きにくくなる。")
                print(f"  応答「{text}」を再生")
                tts.play(wav_bytes)
                print()
    except KeyboardInterrupt:
        print("\n終了します。")
    finally:
        player.close()


def main():
    parser = argparse.ArgumentParser(description="フィラー音を実際に鳴らして確かめる")
    parser.add_argument("--phrases", nargs="+", default=list(DEFAULT_FILLER_PHRASES),
                        help="鳴らすフィラー（複数指定可。直前と同じものを避けて無作為に選ぶ）")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="1つ鳴らし終えてから次を鳴らすまでの間隔（秒）")
    parser.add_argument("--delay", type=float, default=0.4, metavar="SEC",
                        help="認識確定から1つ目を鳴らし始めるまでの待ち（既定0.4秒）")
    parser.add_argument("--reply", type=str, default=SAMPLE_REPLY, metavar="TEXT",
                        help="--simulate で鳴らす応答の文面（既定は感動詞つきの例）")
    parser.add_argument("--compare-interjection", action="store_true",
                        help="--simulate で、感動詞あり／なしの応答を続けて鳴らして聞き比べる")
    parser.add_argument("--simulate", type=float, default=None, metavar="SEC",
                        help="1ターンを再現する。SEC秒後に応答ができたことにして切り替える")
    parser.add_argument("--seed", type=int, default=None,
                        help="フィラー選択の乱数シード。指定すると並びが再現する")
    parser.add_argument("--swap-ratio", type=float, default=0.5,
                        help="子音の差し替え率（フィラーはほぼ影響を受けないが応答には効く）")
    args = parser.parse_args()
    run(args.phrases, args.interval, args.delay, args.simulate, args.reply,
        args.compare_interjection, args.swap_ratio, args.seed)


if __name__ == "__main__":
    main()
