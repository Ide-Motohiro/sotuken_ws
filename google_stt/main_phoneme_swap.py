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
import sys
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

# 発話長は了解性そのものよりも「意図を汲み取るときの疲れにくさ」に効くことが
# 聴取比較で分かっている。応答を短く保つ指示をここで与えるかどうかは未確定のため、
# 現状は main_pseudo.py と同じプロンプトのままにしてある。
SYSTEM_PROMPT = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"


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
    model = GeminiLanguageModel(
        system_instruction=history.system_instruction
    )

    # 6. アプリケーションサービス (ユースケース) の作成
    # VOICEVOX は文章全体からクエリを作る必要があるため use_stream=False
    service = DialogueApplicationService(
        recognizer=recognizer,
        model=model,
        tts=tts,
        history=history,
        use_stream=False
    )

    print(f"準備完了（置換表={args.table} / 差し替え率={args.swap_ratio}）。"
          "話しかけてください。（Ctrl+C で終了）\n")
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
