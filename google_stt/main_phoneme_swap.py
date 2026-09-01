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
import os
import sys
from datetime import datetime
from typing import Callable, Optional
from dotenv import find_dotenv, load_dotenv

# 環境変数の読み込み
load_dotenv(find_dotenv())

from google_stt.domain.models import ConsonantSubstitutionTable, DialogueHistory
from google_stt.domain.application import DialogueApplicationService
from google_stt.infrastructure.google_stt import (
    INPUT_DEVICE_ENV_VAR, GoogleSpeechRecognizer, describe_input_device,
    is_loopback_device, list_input_devices, measure_input_level, resolve_input_device,
)
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.phoneme_swap import PhonemeSwapTTS
from google_stt.infrastructure.voicevox import VoiceVoxTTS, check_server
from google_stt.infrastructure.filler import DEFAULT_FILLER_PHRASES, VoiceVoxFillerPlayer

# 発話長は了解性そのものよりも「意図を汲み取るときの疲れにくさ」に効くことが聴取比較で
# 分かっているため、応答を短く保つ。実測では素の「簡潔に答えて」が中央値80モーラだったのに対し、
# このプロンプトは8モーラに収まる。
#
# 設計上の要点（詳細と測定値は DECISIONS.md の「応答を短く保つシステムプロンプト」）：
#   ・「30モーラ以内」のような数値制約は使わない。LLMはモーラを正確に数えられず、
#     few-shot の例で見せる方が確実に効く
#   ・読点は意味のまとまりごとに打つ（ルール3）。以前は単語ごとに区切っていたが、
#     細かすぎるという判断で分節相当まで粗くした。**「単語ごとに区切る」は短さの制約も
#     兼ねていた**ので、外すだけだと応答が普通の日本語に戻る（8→13モーラ）。
#     「助詞を落とす」を独立させ、例も片言のまま書き換えて短さを保っている。
#     読点2個→1個で 再生 2.53→2.40秒、合成 863→810ms。ただし文末表現は必ず残す。
#     文末は発話の態度を担う手がかりで、疑問の抑揚は子音置換の影響を受けない
#     （語そのものは受ける。「かな」は「タナ」、「なの？」は「ナモ」になる）
#   ・ルール8（覚えているふりの禁止）は「前の会話を覚えているのでは」という疑いから入れた。
#     履歴は毎回空から始まるので保存はされていないが、以前のやり取りを前提にされると
#     モデルが話を合わせて作り話をする（空の履歴で6件中3件。「先週のあの店よかったよね」に
#     「うん、おさけ、おいしかったね」と答えた）。ルール7で「相手の発話に無い言葉を入れて
#     話を進めろ」と縛っているぶん、でっち上げる方向に押していた可能性がある。
#     入れた結果6件中0件になり、通常会話への影響は見られなかった
#   ・冒頭2行の状況説明は実対話のログから入れた。**自分が会話の相手であることを
#     見失う**（「会話が下手」と言われて「だれと話すの」と返した）のと、
#     **時刻を知らないのに推測する**（「もう2時」を深夜と決めつけた）ため。
#     **疑似言語音で喋っていることは教えていない。** 教えると「聞こえてる？」の確認が
#     増えて話が進まなくなるおそれがあり、聞き返しはルール10で既に扱えている
#   ・ルール8（沈黙時に自分から話しかける）は「会話してる感が薄い」という指摘から入れた。
#     ログ37ターンすべてでユーザーが起点だった。認識が時間切れになると
#     DialogueApplicationService が SILENCE_MARKER を履歴へ入れて応答を生成する。
#     ルールを入れないと、履歴が空のときに「だんまり、さみしいな。」と沈黙自体に言及した
#   ・ルール7で「質問」を進め方の選択肢から外している。以前は「自分の話をする／聞く／
#     別の見方を出す」の3択にしていたが、**質問が最も安全に新語を入れられる手段**なので
#     そこに寄り、応答の58%が質問になって尋問のように感じられた（実対話37ターンでも54%）。
#     「3回に1回まで」のように数えさせる書き方は効くが生成が約2秒延びる。
#     手段そのものを「自分の話を足す」に一本化すると、コストゼロで 質問25%／自己開示42%／
#     受け止め29% に散った（n=24）。禁止はしていないので質問は自然に出る
#   ・ルール7（オウム返しの禁止）は実対話の観察から入れた。短さを最優先させると、
#     相手の発話を圧縮して「〜だね」を付けるのが最も安全な出力になり、話が前に進まなくなる。
#     [例] も部分的にオウム返しを教えていたので、話を進める形に差し替えた。
#     実測では新語率の中央値が 60% → 100% に上がり、応答はむしろ短くなった（12→9モーラ）
#   ・応答の文頭に感動詞を置かせない（ルール6）。フィラーも感動詞なので、
#     「えーと」→「うん、…」と感動詞が二重になり、しかも人とは逆順（言い淀み→反応）になる。
#     実際に対話して食い気味・不自然と判断した。態度の手がかりはフィラー側が担う
#   ・矛盾する指示（「短く」かつ「鋭い分析を」等）は無視されるので入れない
#   ・ルール7は、置換表が固定で「同じ内容を言えば必ず同じ音になる」ことへの対処。
#     LLMは自分の声が言葉として届いていないことを知らないため、放っておくと聞き返しに
#     対して同じ内容を返し、同じ雑音がもう一度鳴るだけになる（DECISIONS.md の問題1）。
#     求めるのは「言い換え」ではなく「言い直し」。話題を変えて別のことを言い出すのではなく、
#     人が聞き返されたときのように同じ内容を段階的に削っていく
#   ・語彙の自然さ（「つかれている」対「つかれてる」等）は調整しない。疑似言語音では
#     聞き手に届かない一方、ルールを足すぶん生成が遅くなり他のルールの遵守も緩むため。
#     教科書的な言い回しは「日本語を学習中」という設定とも整合する
#
# **この文面を変えたら応答の長さと生成時間を測り直すこと。** 制約を足すほど生成が遅くなる。
#: 相手が黙ったまま時間切れになったとき、ユーザー発話の代わりに履歴へ入れる印。
#: ルール8がこの文字列を見て「自分から話しかける」に切り替わる。
#: ログには user としてそのまま残るので、こちらから話しかけたターンは後から見分けられる。
SILENCE_MARKER = "（相手が黙っている）"

SYSTEM_PROMPT_TEMPLATE = """あなたは日本語を学習中の、聡明だが言葉が達者でない話し相手です。
（状況：あなたは声で相手と話している。相手が「君」「あなた」と言うのはあなたのこと。いまは{now}。）

[ルール]
1. 返答は1文。長くても2文。
2. 前置き・挨拶・同調（「いい質問ですね」「なるほど」等）は禁止。核心だけを言う。
3. 助詞を落として、ぶっきらぼうに話す。ひらがなで書ける語はひらがなにする。
   読点は意味のまとまりごとに、1文に2個まで打つ。単語ごとには区切らない。
4. ただし文末は「〜だね」「〜かな」「〜だよ」「〜なの？」を必ず残す。丁寧語は使わない。
5. 説明・理由・列挙はしない。短さを最優先する。
6. 文頭に感動詞（あー、うん、へえ、おお）を置かない。いきなり内容から始める。
7. 相手が言った言葉をそのまま返さない。自分の話（好き嫌い・したこと・思ったこと）を足して、
   話を前に進める。質問だけで返さない。
8. この会話でまだ出ていないことは知らない。「さっきの」「この前」「昨日」のように以前のやり取りを前提にされても、知っているふりをしない。何の話か素直に聞く。
9. 「え？」「なんて？」「もう一回」のように聞き返されたら、話題を変えない。直前の自分の発言を、そのまま短く言い直す。聞き返されるたびに短くしていき、最後は一番大事な単語だけにする。文末に「って」を付けてもよい。

[例]
ユーザー「今日はいい天気だね」→「そとでごはん、たべたいかな」
ユーザー「なんかお昼寝がしたい気分」→「わたしも、ひるねすきだよ」
ユーザー「昼寝って起きれなくない？」→「ねむいとき、なにするかな」
ユーザー「週末どこか行く？」→「うみいきたい、かな」
ユーザー「テスト受かったよ」→「なにべんきょう、したの？」
ユーザー「これ食べていい？」→「だめ、それわたしの」

[聞き返しの例]
「そとでごはん、たべたいかな」と言ったあとに
  ユーザー「え？なんて？」→「そとでごはん、って」
  ユーザー「え？」→「そと」
「うみいきたい、かな」と言ったあとに
  ユーザー「もう一回言って」→「うみ、って」
  ユーザー「え？」→「うみ」{silence_section}"""


#: 沈黙時にこちらから話しかける機能を使うときだけ足す節。
#: **無効なときは足さないこと。** マーカーの文字列をプロンプトに書くと、モデルが
#: それをそのまま応答として出力することがある（実測で1回発生）。
SILENCE_SECTION_TEMPLATE = """

[相手が黙っているとき]
「{marker}」と書かれたら、相手は何も言っていない。自分から新しい話しかけをする。
黙っていることには触れない。"""


def build_system_prompt(now: Optional[datetime] = None,
                        silence_marker: Optional[str] = None) -> str:
    """現在時刻を埋めたシステムプロンプトを返す。

    起動時に一度だけ呼ぶ。会話の途中で時刻を更新はしない（履歴に残る文言が
    ターンごとに変わると、モデルから見た文脈が不安定になるため）。

    silence_marker を渡したときだけ、沈黙時に話しかける節を足す。
    """
    stamp = (now or datetime.now()).strftime("%Y年%m月%d日 %H時%M分")
    section = ("" if silence_marker is None
               else SILENCE_SECTION_TEMPLATE.format(marker=silence_marker))
    return SYSTEM_PROMPT_TEMPLATE.format(now=stamp, silence_section=section)


def make_turn_logger(
    service: DialogueApplicationService, log_path: Optional[str],
) -> Callable[[str, str], None]:
    """1ターンごとに認識テキストと応答テキストを表示し、指定があればJSONLで追記する。

    疑似言語音では応答が語彙として届かないため、**認識結果を残しておかないと会話が脱線した
    原因を後から追えない**。ASRが雑音や疑似言語音を拾うとLLMがそこから意味をでっち上げ、
    自然に見える応答を返すことがある（DECISIONS.md の問題2）。
    """
    def log_turn(user_text: str, reply_text: str) -> None:
        print(f"  認識: {user_text}")
        print(f"  応答: {reply_text}")
        turn = service.last_turn_timing
        if turn is not None and turn.time_to_response_sec is not None:
            print(f"  応答まで {turn.time_to_response_sec:.2f}秒"
                  f"（終端判定 {turn.endpoint_wait_sec:.2f} / 生成 {turn.generation_sec:.2f} / "
                  f"合成 {turn.time_to_first_sound_sec:.2f}"
                  + (f" / フィラー待ち {turn.filler_stop_delay_sec:.2f}"
                     if turn.filler_stop_delay_sec else "") + "）")
        if log_path is None:
            return
        def rounded(value):
            return round(value, 3) if value is not None else None

        synthesis = service.tts.last_timing
        turn = service.last_turn_timing
        recognizer = service.recognizer
        record = {
            "time": datetime.now().isoformat(timespec="milliseconds"),
            "user": user_text,
            "reply": reply_text,
            # 合成の内訳（VOICEVOX の2段階呼び出しに対応）
            "query_sec": rounded(synthesis.query_sec) if synthesis else None,
            "synthesis_sec": rounded(synthesis.synthesis_sec) if synthesis else None,
            # 発話終端からの通し。**endpoint_wait_sec の起点は中間結果が変化しなくなった
            # 時刻であって、ユーザーが口を閉じた時刻ではない**（認識結果の遅れは未計測）
            "endpoint_wait_sec": rounded(turn.endpoint_wait_sec) if turn else None,
            "generation_sec": rounded(turn.generation_sec) if turn else None,
            "time_to_first_sound_sec": rounded(turn.time_to_first_sound_sec) if turn else None,
            "playback_sec": rounded(turn.playback_sec) if turn else None,
            # フィラーがまだ鳴っていて応答を待たせた時間。フィラーあり条件だけ応答が
            # 遅くなると交絡になるので、推定ではなく実測を残す（無し条件では null）
            "filler_stop_delay_sec": rounded(turn.filler_stop_delay_sec) if turn else None,
            # 上を足したもの。体感される「応答までの間」に相当する
            "time_to_response_sec": rounded(turn.time_to_response_sec) if turn else None,
            # 認識結果の信頼度。Google STT は最終結果にしか入れないと思われるため、
            # 中間結果で確定した経路（finalized_by_service=false）では 0.0 になる想定
            # 差し替え率は聞き取りやすさのつまみ。条件を変えて話した記録が混ざるので毎ターン残す
            "swap_ratio": service.tts.substitution_table.swap_ratio,
            # このターンの時点で履歴に積まれている発話数。**起動直後の1ターン目は必ず2になる**
            # （このターンのユーザー発話＋応答）。履歴は保存されず毎回空から始まることの確認用
            "history_turns": len(service.history.get_messages()),
            "confidence": recognizer.last_confidence,
            "finalized_by_service": getattr(recognizer, "last_finalized_by_service", None),
        }
        # 1ターンごとに追記して flush する（途中で落ちてもそこまでのログが残るように）
        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return log_turn


def main() -> None:
    parser = argparse.ArgumentParser(description="疑似言語音（音素差し替え方式）の対話システム")
    parser.add_argument(
        "--swap-ratio", type=float, default=0.33,
        help="子音を差し替えるモーラの割合（0.0〜1.0）。大きいほど分かりにくくなる。既定は0.33。**初めて聞く人には0.5でも聞き取れない**という指摘で下げた",
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
        "--check-mic", action="store_true",
        help="使うマイクが実際に声を拾うかを5秒間の録音で確かめて終了する。"
             "選択を間違えると認識が例外も出さずに返ってこないので、実験前に必ず通すこと",
    )
    parser.add_argument(
        "--push-to-talk", action="store_true",
        help="スペースを押している間だけマイクを開く。離した時点で発話終端とする。"
             "終端判定の待ち（約1秒）が消え、押していない間は周りの声を拾わない。"
             "騒がしい場所での実演向け。--endpoint-wait 系は使われなくなる",
    )
    parser.add_argument(
        "--endpoint-wait", type=float, default=0.4, metavar="SEC",
        help="発話終端とみなすまでの無音時間（既定0.4秒）。"
             "**フィラーが鳴り始める前の区間なので、伸ばすと隠せない無音がそのまま増える**",
    )
    parser.add_argument(
        "--endpoint-wait-continuing", type=float, default=1.0, metavar="SEC",
        help="まだ続きそうな終わり方（助詞・接続助詞・言いよどみ）のときに待つ時間（既定1.0秒）。"
             "「昨日は……」で切られるのを防ぐ。--endpoint-wait 以上にすること",
    )
    parser.add_argument(
        "--idle-timeout", type=float, default=0.0, metavar="SEC",
        help="相手が黙ったままこの秒数が過ぎたら、こちらから話しかける。"
             "**既定は0（無効）**で、相手の発話を待ち続ける。8 くらいを渡すと有効になる",
    )
    parser.add_argument(
        "--no-filler", action="store_true",
        help="応答生成の待機中にフィラーを鳴らさない（実験条件「フィラーの有無」の切り替え）",
    )
    parser.add_argument(
        "--filler-interval", type=float, default=2.0, metavar="SEC",
        help="応答生成が長引いたときに2つ目のフィラーを鳴らすまでの固定間隔（既定2.0秒）",
    )
    parser.add_argument(
        "--filler-delay", type=float, default=0.4, metavar="SEC",
        help="認識が確定してから1つ目のフィラーを鳴らし始めるまでの待ち（既定0.4秒）。"
             "0にすると確定と同時に鳴り、食い気味に聞こえる。"
             "鳴っているフィラーは最後まで鳴らすので、大きくするとその分応答が遅れる"
             "（遅れた秒数は --log の filler_stop_delay_sec に残る）",
    )
    parser.add_argument(
        "--filler-seed", type=int, default=None, metavar="N",
        help="フィラー選択の乱数シード。既定は実行ごとに変わる。指定すると並びが再現する（デバッグ用）",
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

    # マイクの選択を誤ると recognize_once() が例外も出さずに返ってこないだけになる。
    # 原因の切り分けに時間を取られるので、対話を回す前にここで音量を見る。
    if args.check_mic:
        index = resolve_input_device(args.input_device)
        print(f"使用デバイス: {describe_input_device(index)}")
        if is_loopback_device(index):
            print("警告: ループバック（システムの再生音を拾う）に見える。声は入らない",
                  file=sys.stderr)
        print()
        print("5秒間、いつもの声量で話してください。")
        blocks, median, peak = measure_input_level(
            index, on_tick=lambda r: print(f"  録音中... 残り{r:.0f}秒", flush=True))
        if blocks == 0:
            print("音声ブロックが1つも届かなかった。デバイスを開けていない", file=sys.stderr)
            sys.exit(1)
        print()
        print(f"ブロック数 {blocks} / RMS 中央値 {median:.1f} / 最大 {peak:.1f}"
              f"（16bit の最大振幅は 32767）")
        # 目安：静かな部屋の無音で中央値10前後、通常の発話では中央値が数百に乗る
        if peak < 50:
            print("→ ほぼ無音。マイクが違うか、ミュートか、権限が無い")
        elif median < 100:
            print("→ 物音は拾っているが発話が入っていない。声が届いていないか感度が低い")
        else:
            print("→ 十分拾えている")
        return

    # VOICEVOX が起動していないと、最初の合成（フィラーの事前合成）で長大な接続エラーが出る。
    # 何が起きたか読み取れないので、ここで1行にして止める。
    problem = check_server(VoiceVoxTTS.DEFAULT_URL)
    if problem is not None:
        print(f"エラー: {problem}", file=sys.stderr)
        sys.exit(1)

    print("システム初期化中...")

    # 1. 履歴管理 (ドメイン) の作成
    silence_marker = SILENCE_MARKER if args.idle_timeout > 0 else None
    history = DialogueHistory(system_instruction=build_system_prompt(
        silence_marker=silence_marker))

    # 2. 子音置換ルール (ドメイン) の作成
    if args.table == "articulatory":
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=args.swap_ratio)
    else:
        table = ConsonantSubstitutionTable.distant(swap_ratio=args.swap_ratio)

    # 3. 疑似言語音 TTS (インフラ) の作成
    tts = PhonemeSwapTTS(substitution_table=table)

    # 4. フィラー (インフラ) の作成。--no-filler なら None（＝フィラー無し条件）
    #    停止点を「合成後・再生前」に置くため on_before_playback に配線する。
    #    ここを配線しないと、応答テキストが返った時点で止まって合成の約1.1秒が無音になる。
    filler = None
    if not args.no_filler:
        # 合成だけ素の VoiceVoxTTS で行い、フィラーには子音置換をかけない。
        # 再生中の is_speaking は本体の tts に立てるので、エコー防止はそのまま働く。
        filler = VoiceVoxFillerPlayer(
            tts, synthesizer=VoiceVoxTTS(url=tts.url, speaker_id=tts.speaker_id),
            phrases=DEFAULT_FILLER_PHRASES,
            repeat_interval_sec=args.filler_interval,
            initial_delay_sec=args.filler_delay, seed=args.filler_seed)
        tts.on_before_playback = filler.stop

    # 5. ASR (インフラ) の作成 (再生中は録音をスキップしてエコーを防止)
    recognizer = GoogleSpeechRecognizer(
        is_speaking_fn=lambda: tts.is_speaking,
        device=args.input_device,
        stability_duration=args.endpoint_wait,
        continuation_duration=args.endpoint_wait_continuing,
        silence_timeout_sec=args.idle_timeout if args.idle_timeout > 0 else None,
        push_to_talk=args.push_to_talk,
    )

    # 6. LLM (インフラ) の作成
    # 疑似言語音では語彙が聞き手に届かないため内容の精度が効きにくい一方、レイテンシは
    # 研究の従属変数そのもの。そのため既定では思考を切る（-1 でAPI既定に戻せる）
    model = GeminiLanguageModel(
        system_instruction=history.system_instruction,
        thinking_budget=None if args.thinking_budget < 0 else args.thinking_budget,
    )

    # 7. アプリケーションサービス (ユースケース) の作成
    # VOICEVOX は文章全体からクエリを作る必要があるため use_stream=False
    service = DialogueApplicationService(
        recognizer=recognizer,
        model=model,
        tts=tts,
        history=history,
        use_stream=False,
        filler=filler,
        # 相手が黙ったままなら、こちらから話しかける（--idle-timeout 0 で無効）
        silence_marker=silence_marker,
    )
    # ログは service が束ねた計測値を読むので、service を作ってから配線する
    service.on_turn = make_turn_logger(service, args.log)

    print("こちらから話しかける: "
          + (f"{args.idle_timeout}秒の沈黙で" if args.idle_timeout > 0 else "しない"))
    if args.push_to_talk:
        print("終端判定: 手押しトリガー（スペースを押している間だけ録音）")
    else:
        print(f"終端判定: {args.endpoint_wait}秒"
              f"（続きそうな終わり方なら {args.endpoint_wait_continuing}秒）")
    print(f"準備完了（置換表={args.table} / 差し替え率={args.swap_ratio} / "
          f"thinking={args.thinking_budget} / "
          f"フィラー={'なし' if args.no_filler else f'あり（{args.filler_delay}秒後）'}）。")
    if args.log:
        print(f"ログ: {args.log}")
    print("話しかけてください。（Ctrl+C で終了）\n")
    try:
        while True:
            try:
                service.run_once()
            except KeyboardInterrupt:
                print("\n終了します。")
                break
            except Exception as e:
                print(f"\nエラーが発生しました: {e}", file=sys.stderr)
    finally:
        # 事前合成したフィラーの一時ファイルを片付ける
        if filler is not None:
            filler.close()


if __name__ == "__main__":
    main()
