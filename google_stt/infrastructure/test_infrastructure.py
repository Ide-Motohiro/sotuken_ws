import unittest
import os
import requests
import sounddevice as sd
from google_stt.domain.models import DialogueHistory, FillerSequence, PseudoVoicePitchMapper
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.voicevox import VoiceVoxTTS
from google_stt.infrastructure.pseudo_voice import PseudoVoiceTTS
from google_stt.infrastructure.google_stt import GoogleSpeechRecognizer

# 環境と依存サーバーのチェック
def has_google_credentials() -> bool:
    # デフォルトクレデンシャルまたは環境変数が設定されているか簡易チェック
    return (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
        or os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    )

def is_voicevox_running() -> bool:
    try:
        # VOICEVOXが起動しているか確認
        response = requests.get("http://localhost:50021/speakers", timeout=0.5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def has_audio_output() -> bool:
    try:
        # 有効な出力デバイスがあるかチェック
        devices = sd.query_devices()
        for dev in devices:
            if dev.get("max_output_channels", 0) > 0:
                return True
        return False
    except Exception:
        return False


class TestGeminiLanguageModelIntegration(unittest.TestCase):
    @unittest.skipUnless(has_google_credentials(), "Google Cloud credentials not configured")
    def test_gemini_generate_reply(self):
        """Geminiによる一括応答生成の疎通テスト"""
        llm = GeminiLanguageModel(system_instruction="「はい」とだけ返答してください。")
        history = DialogueHistory()
        history.add_user_message("テスト疎通")
        
        reply = llm.generate_reply(history)
        self.assertTrue(len(reply) > 0)
        self.assertEqual(reply, "はい")

    @unittest.skipUnless(has_google_credentials(), "Google Cloud credentials not configured")
    def test_gemini_generate_reply_stream(self):
        """Geminiによるストリーミング応答生成の疎通テスト"""
        llm = GeminiLanguageModel(system_instruction="「はい」とだけ返答してください。")
        history = DialogueHistory()
        history.add_user_message("テスト疎通")
        
        stream = llm.generate_reply_stream(history)
        chars = list(stream)
        reply = "".join(chars)
        self.assertTrue(len(reply) > 0)
        self.assertEqual(reply, "はい")


class TestVoiceVoxTTSIntegration(unittest.TestCase):
    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running on localhost:50021")
    def test_voicevox_speak(self):
        """VOICEVOX音声合成と再生処理の疎通テスト"""
        tts = VoiceVoxTTS(url="http://localhost:50021", speaker_id=3)
        # winsound等での再生がエラーなく完遂されるかテスト
        try:
            tts.speak("テスト")
        except Exception as e:
            self.fail(f"VOICEVOX speak failed with exception: {e}")


class TestPseudoVoiceTTSIntegration(unittest.TestCase):
    @unittest.skipUnless(has_audio_output(), "No audio output devices available")
    def test_pseudo_voice_speak(self):
        """サイン波ビープ疑似音声とフィラー再生スレッドの動作テスト"""
        mapper = PseudoVoicePitchMapper(scale_frequencies=[440.0, 520.0], skip_characters=[" "])
        seq = FillerSequence(["あ", "い"])
        
        tts = PseudoVoiceTTS(
            pitch_mapper=mapper,
            filler_sequence=seq,
            volume=0.01,  # テストのため極小音量
            char_duration=0.01,
            filler_duration=0.01,
            filler_pause=0.01
        )
        
        try:
            # 一括再生のテスト
            tts.speak("あい")
            
            # ストリーミング＆フィラー非同期再生スレッドのテスト
            tts.speak_stream(iter(["あ", "い"]))
        except Exception as e:
            self.fail(f"PseudoVoiceTTS failed with exception: {e}")


class TestGoogleSpeechRecognizerIntegration(unittest.TestCase):
    @unittest.skipUnless(has_google_credentials(), "Google Cloud credentials not configured")
    def test_google_stt_initialization(self):
        """GoogleSpeechRecognizerの初期化接続テスト"""
        try:
            recognizer = GoogleSpeechRecognizer()
            self.assertIsNotNone(recognizer._stt_client)
        except Exception as e:
            self.fail(f"GoogleSpeechRecognizer initialization failed: {e}")


if __name__ == "__main__":
    unittest.main()
