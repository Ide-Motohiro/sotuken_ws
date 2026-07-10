import unittest
from google_stt.domain.models import DialogueHistory
from google_stt.domain.mocks import (
    MockSpeechRecognizer,
    MockLanguageModel,
    MockTextToSpeech,
)

class TestMockSpeechRecognizer(unittest.TestCase):
    def test_recognize_once_sequential(self):
        """MockSpeechRecognizerが設定したテキストを順番に返すこと"""
        recognizer = MockSpeechRecognizer(["こんにちは", "テスト"])
        self.assertEqual(recognizer.recognize_once(), "こんにちは")
        self.assertEqual(recognizer.recognize_once(), "テスト")

    def test_recognize_once_out_of_bounds(self):
        """設定テキストを使い果たした後は空文字列を返すこと"""
        recognizer = MockSpeechRecognizer(["こんにちは"])
        self.assertEqual(recognizer.recognize_once(), "こんにちは")
        self.assertEqual(recognizer.recognize_once(), "")
        self.assertEqual(recognizer.recognize_once(), "")


class TestMockLanguageModel(unittest.TestCase):
    def test_generate_reply(self):
        """generate_replyが一括で設定された返答を返すこと"""
        llm = MockLanguageModel(reply="こんにちは、私はAIです。")
        history = DialogueHistory()
        history.add_user_message("ハロー")
        
        reply = llm.generate_reply(history)
        self.assertEqual(reply, "こんにちは、私はAIです。")

    def test_generate_reply_stream(self):
        """generate_reply_streamが文字単位でストリーミング返答を返すこと"""
        llm = MockLanguageModel(reply="どうも")
        history = DialogueHistory()
        history.add_user_message("ハロー")
        
        stream = llm.generate_reply_stream(history)
        chars = list(stream)
        self.assertEqual(chars, ["ど", "う", "も"])


class TestMockTextToSpeech(unittest.TestCase):
    def test_speak(self):
        """speakが正しく再生テキストを記録すること"""
        tts = MockTextToSpeech()
        tts.speak("テスト発話")
        self.assertEqual(tts.spoken_texts, ["テスト発話"])

    def test_speak_stream(self):
        """speak_streamがストリームを結合して記録すること"""
        tts = MockTextToSpeech()
        text_stream = iter(["え", "ー", "と", "、", "こんにちは"])
        tts.speak_stream(text_stream)
        self.assertEqual(tts.spoken_texts, ["えーと、こんにちは"])


if __name__ == "__main__":
    unittest.main()
