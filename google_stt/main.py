import sys
from dotenv import find_dotenv, load_dotenv

# 環境変数の読み込み
load_dotenv(find_dotenv())

from google_stt.domain.models import DialogueHistory
from google_stt.domain.application import DialogueApplicationService
from google_stt.infrastructure.google_stt import GoogleSpeechRecognizer
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.voicevox import VoiceVoxTTS

def main():
    print("システム初期化中...")
    
    # 1. 履歴管理 (ドメイン) の作成
    system_prompt = "あなたは親しみやすいAIアシスタントです。簡潔に日本語で答えてください。"
    history = DialogueHistory(system_instruction=system_prompt)
    
    # 2. VOICEVOX TTS (インフラ) の作成
    tts = VoiceVoxTTS(url="http://localhost:50021", speaker_id=3)
    
    # 3. ASR (インフラ) の作成 (TTS再生中は録音をスキップしてエコーを防止)
    recognizer = GoogleSpeechRecognizer(
        is_speaking_fn=lambda: tts.is_speaking
    )
    
    # 4. LLM (インフラ) の作成
    model = GeminiLanguageModel(
        system_instruction=history.system_instruction
    )
    
    # 5. アプリケーションサービス (ユースケース) の作成
    # VOICEVOX は一括再生するため use_stream=False
    service = DialogueApplicationService(
        recognizer=recognizer,
        model=model,
        tts=tts,
        history=history,
        use_stream=False
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
