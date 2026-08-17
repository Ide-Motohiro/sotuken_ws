"""疑似言語音（音素差し替え方式）版のエントリポイント。

Google Cloud STT + Gemini 2.5 Flash + VOICEVOX（合成前クエリの子音を差し替え）。
サイン波版（main_pseudo.py）と違い、VOICEVOX 自身が合成するので音質は素のまま、
韻律・リズム・発話長も元の応答のまま残り、語彙的な意味だけが消える。

使い方：
    python -m google_stt.main_phoneme_swap
    python -m google_stt.main_phoneme_swap --swap-ratio 1.0   # 全モーラ差し替え（より分かりにくい）
    python -m google_stt.main_phoneme_swap --table distant    # 比較対照用の「遠い置換」
"""
import argparse
import json
import sys
from datetime import datetime
from typing import Callable, Optional
from dotenv import find_dotenv, load_dotenv

# 環境変数の読み込み
load_dotenv(find_dotenv())

from google_stt.domain.models import ConsonantSubstitutionTable, DialogueHistory
from google_stt.domain.application import DialogueApplicationService
from google_stt.infrastructure.google_stt import (
    INPUT_DEVICE_ENV_VAR, GoogleSpeechRecognizer, list_input_devices,
)
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.phoneme_swap import PhonemeSwapTTS

# 発話長は了解性そのものよりも「意図を汲み取るときの疲れにくさ」に効くことが聴取比較で
# 分かっているため、応答を短く保つ。実測では素の「簡潔に答えて」が中央値80モーラだったのに対し、
# このプロンプトは8モーラに収まる。
#
# 設計上の要点（詳細と測定値は TODO.md の項目2）：
#   ・「30モーラ以内」のような数値制約は使わない。LLMはモーラを正確に数えられず、
#     few-shot の例で見せる方が確実に効く
#   ・読点で区切るのは聞き取りやすさのため（聞き比べで判断）。ただし文末表現は必ず残す。
#     文末は発話の態度を担う唯一の手がかりで、しかも母音・撥音に偏っていて子音置換で壊れない
#   ・矛盾する指示（「短く」かつ「鋭い分析を」等）は無視されるので入れない
#   ・ルール7は、置換表が固定で「同じ内容を言えば必ず同じ音になる」ことへの対処。
#     LLMは自分の声が言葉として届いていないことを知らないため、放っておくと聞き返しに
#     対して同じ内容を返し、同じ雑音がもう一度鳴るだけになる（TODO.md 項目3の問題1）。
#     求めるのは「言い換え」ではなく「言い直し」。話題を変えて別のことを言い出すのではなく、
#     人が聞き返されたときのように同じ内容を段階的に削っていく
#   ・語彙の自然さ（「つかれている」対「つかれてる」等）は調整しない。疑似言語音では
#     聞き手に届かない一方、ルールを足すぶん生成が遅くなり他のルールの遵守も緩むため。
#     教科書的な言い回しは「日本語を学習中」という設定とも整合する
#
# **この文面を変えたら応答の長さと生成時間を測り直すこと。** 制約を足すほど生成が遅くなる。
SYSTEM_PROMPT = """あなたは日本語を学習中の、聡明だが言葉が達者でない話し相手です。

[ルール]
1. 返答は1文。長くても2文。
2. 前置き・挨拶・同調（「いい質問ですね」「なるほど」等）は禁止。核心だけを言う。
3. 文頭に感動詞（あー、うん、へえ、おお、うわ、えー）を1つ置く。
4. 読点で単語ごとに区切り、ぶっきらぼうに話す。助詞は減らす。
5. ただし文末は「〜だね」「〜かな」「〜だよ」「〜なの？」を必ず残す。丁寧語は使わない。
6. 説明・理由・列挙はしない。短さを最優先する。
7. 「え？」「なんて？」「もう一回」のように聞き返されたら、話題を変えない。直前の自分の発言を、そのまま短く言い直す。聞き返されるたびに短くしていき、最後は一番大事な単語だけにする。感動詞は付けず、文末に「って」を付けてもよい。

[例]
ユーザー「今日はいい天気だね」→「うん、きもち、いいね」
ユーザー「週末どこか行く？」→「あー、まだ、決めてない、かな」
ユーザー「これ食べていい？」→「だめ、それ、わたしの」
ユーザー「テスト受かったよ」→「おお、やった、ね！」

[聞き返しの例]
「うん、きもち、いいね」と言ったあとに
  ユーザー「え？なんて？」→「てんき、いいね、って」
  ユーザー「え？」→「てんき」
「あー、まだ、決めてない、かな」と言ったあとに
  ユーザー「もう一回言って」→「よてい、ない、って」
  ユーザー「え？」→「よてい、ない」"""


def make_turn_logger(tts: PhonemeSwapTTS, log_path: Optional[str]) -> Callable[[str, str], None]:
    """1ターンごとに認識テキストと応答テキストを表示し、指定があればJSONLで追記する。

    疑似言語音では応答が語彙として届かないため、**認識結果を残しておかないと会話が脱線した
    原因を後から追えない**。ASRが雑音や疑似言語音を拾うとLLMがそこから意味をでっち上げ、
    自然に見える応答を返すことがある（TODO.md 項目3の問題2）。
    """
    def log_turn(user_text: str, reply_text: str) -> None:
        print(f"  認識: {user_text}")
        print(f"  応答: {reply_text}")
        if log_path is None:
            return
        timing = tts.last_timing
        record = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "user": user_text,
            "reply": reply_text,
            "query_sec": round(timing.query_sec, 3) if timing else None,
            "synthesis_sec": round(timing.synthesis_sec, 3) if timing else None,
            "playback_sec": round(timing.playback_sec, 3) if timing else None,
        }
        # 1ターンごとに追記して flush する（途中で落ちてもそこまでのログが残るように）
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return log_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="疑似言語音（音素差し替え方式）の対話システム")
    parser.add_argument(
        "--swap-ratio", type=float, default=0.5,
        help="子音を差し替えるモーラの割合（0.0〜1.0）。大きいほど分かりにくくなる。既定は聴取比較で選んだ0.5",
    )
    parser.add_argument(
        "--table", choices=["articulatory", "distant"], default="articulatory",
        help="置換表。articulatory=調音的に近い音へ（既定）、distant=遠い音へ（比較対照用）",
    )
    parser.add_argument(
        "--input-device", type=str, default=None,
        help=f"マイク入力デバイス。番号（7）でも名前の一部（Microphone）でもよい。"
             f"未指定なら環境変数 {INPUT_DEVICE_ENV_VAR}、それも無ければOSの既定",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="利用できる入力デバイスを一覧表示して終了する",
    )
    parser.add_argument(
        "--log", type=str, default=None, metavar="PATH",
        help="1ターンごとの認識テキスト・応答テキスト・合成時間をJSONLで追記する。実験時は必ず指定すること",
    )
    parser.add_argument(
        "--thinking-budget", type=int, default=0,
        help="Geminiの思考トークン上限。既定0（思考を無効化）。実測で応答生成が約5.7倍速くなる。"
             "-1 を渡すとAPIの既定（思考する）に戻す",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("利用できる入力デバイス:")
        for index, name in list_input_devices():
            print(f"  {index:>3}  {name}")
        print(f"\n.env に「{INPUT_DEVICE_ENV_VAR}=Microphone Array」のように書けば既定にできる。")
        return

    print("システム初期化中...")

    # 1. 履歴管理 (ドメイン) の作成
    history = DialogueHistory(system_instruction=SYSTEM_PROMPT)

    # 2. 子音置換ルール (ドメイン) の作成
    if args.table == "articulatory":
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=args.swap_ratio)
    else:
        table = ConsonantSubstitutionTable.distant(swap_ratio=args.swap_ratio)

    # 3. 疑似言語音 TTS (インフラ) の作成
    tts = PhonemeSwapTTS(substitution_table=table)

    # 4. ASR (インフラ) の作成 (再生中は録音をスキップしてエコーを防止)
    recognizer = GoogleSpeechRecognizer(
        is_speaking_fn=lambda: tts.is_speaking,
        device=args.input_device,
    )

    # 5. LLM (インフラ) の作成
    # 疑似言語音では語彙が聞き手に届かないため内容の精度が効きにくい一方、レイテンシは
    # 研究の従属変数そのもの。そのため既定では思考を切る（-1 でAPI既定に戻せる）
    model = GeminiLanguageModel(
        system_instruction=history.system_instruction,
        thinking_budget=None if args.thinking_budget < 0 else args.thinking_budget,
    )

    # 6. アプリケーションサービス (ユースケース) の作成
    # VOICEVOX は文章全体からクエリを作る必要があるため use_stream=False
    service = DialogueApplicationService(
        recognizer=recognizer,
        model=model,
        tts=tts,
        history=history,
        use_stream=False,
        on_turn=make_turn_logger(tts, args.log),
    )

    print(f"準備完了（置換表={args.table} / 差し替え率={args.swap_ratio} / "
          f"thinking={args.thinking_budget}）。")
    if args.log:
        print(f"ログ: {args.log}")
    print("話しかけてください。（Ctrl+C で終了）\n")
    while True:
        try:
            service.run_once()
        except KeyboardInterrupt:
            print("\n終了します。")
            break
        except Exception as e:
            print(f"\nエラーが発生しました: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
