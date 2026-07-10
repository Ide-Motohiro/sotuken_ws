import unittest
from google_stt.domain.models import DialogueHistory, Role
from google_stt.domain.mocks import (
    MockSpeechRecognizer,
    MockLanguageModel,
    MockTextToSpeech,
)
from google_stt.domain.application import DialogueApplicationService

class TestDialogueApplicationService(unittest.TestCase):
    def test_run_once_batch_mode(self):
        """一括再生モード (use_stream=False) の正常系テスト"""
        recognizer = MockSpeechRecognizer(["こんにちは"])
        llm = MockLanguageModel(reply="こんにちは、私はAIです。")
        tts = MockTextToSpeech()
        history = DialogueHistory()

        service = DialogueApplicationService(
            recognizer=recognizer,
            model=llm,
            tts=tts,
            history=history,
            use_stream=False
        )

        service.run_once()

        # TTSが正しく呼ばれたか
        self.assertEqual(tts.spoken_texts, ["こんにちは、私はAIです。"])

        # 会話履歴が正しく更新されたか
        messages = history.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, Role.USER)
        self.assertEqual(messages[0].text, "こんにちは")
        self.assertEqual(messages[1].role, Role.MODEL)
        self.assertEqual(messages[1].text, "こんにちは、私はAIです。")

    def test_run_once_stream_mode(self):
        """ストリーミング再生モード (use_stream=True) の正常系テスト"""
        recognizer = MockSpeechRecognizer(["テスト"])
        llm = MockLanguageModel(reply="応答します")
        tts = MockTextToSpeech()
        history = DialogueHistory()

        service = DialogueApplicationService(
            recognizer=recognizer,
            model=llm,
            tts=tts,
            history=history,
            use_stream=True
        )

        service.run_once()

        # TTSにイテレータが渡され、すべて消費・再生されたか
        self.assertEqual(tts.spoken_texts, ["応答します"])

        # 会話履歴にストリーミングから結合されたモデルの返答が記録されたか
        messages = history.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, Role.USER)
        self.assertEqual(messages[0].text, "テスト")
        self.assertEqual(messages[1].role, Role.MODEL)
        self.assertEqual(messages[1].text, "応答します")

    def test_run_once_empty_input_ignored(self):
        """音声認識入力が空の場合は対話処理をスキップすること"""
        recognizer = MockSpeechRecognizer([""])
        llm = MockLanguageModel(reply="応答しません")
        tts = MockTextToSpeech()
        history = DialogueHistory()

        service = DialogueApplicationService(
            recognizer=recognizer,
            model=llm,
            tts=tts,
            history=history,
            use_stream=False
        )

        service.run_once()

        # TTSは呼び出されていないこと
        self.assertEqual(len(tts.spoken_texts), 0)

        # 履歴も空のままであること
        self.assertEqual(len(history.get_messages()), 0)


if __name__ == "__main__":
    unittest.main()
