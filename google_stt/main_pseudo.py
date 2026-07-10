import sys
from dotenv import find_dotenv, load_dotenv

# 環境変数の読み込み
load_dotenv(find_dotenv())

from google_stt.domain.models import DialogueHistory, FillerSequence, PseudoVoicePitchMapper
from google_stt.domain.application import DialogueApplicationService
from google_stt.infrastructure.google_stt import GoogleSpeechRecognizer
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.pseudo_voice import PseudoVoiceTTS

def main():
    print("システム初期化中...")
    
    # 1. 履歴管理 (ドメイン) の作成
    system_prompt = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"
    history = DialogueHistory(system_instruction=system_prompt)
    
    # 2. 疑似音声用ドメインルールの作成
    pitch_mapper = PseudoVoicePitchMapper(
        scale_frequencies=[261.6, 329.6, 392.0, 523.3, 659.3, 784.0, 1046.5],
        skip_characters=[" ", "　", "\n", "\r", "\t"]
    )
    filler_seq = FillerSequence(["え", "ー", "と"])
    
    # 3. 疑似音声 TTS (インフラ) の作成
    tts = PseudoVoiceTTS(
        pitch_mapper=pitch_mapper,
        filler_sequence=filler_seq
    )
    
    # 4. ASR (インフラ) の作成 (ビープ再生中は録音をスキップしてエコーを防止)
    recognizer = GoogleSpeechRecognizer(
        is_speaking_fn=lambda: tts.is_speaking
    )
    
    # 5. LLM (インフラ) の作成
    model = GeminiLanguageModel(
        system_instruction=history.system_instruction
    )
    
    # 6. アプリケーションサービス (ユースケース) の作成
    # サイン波電子疑似音声はストリーミング再生するため use_stream=True
    service = DialogueApplicationService(
        recognizer=recognizer,
        model=model,
        tts=tts,
        history=history,
        use_stream=True
    )
    
    print("準備完了。話しかけてください。（Ctrl+C で終了）\n")
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
