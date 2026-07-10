import unittest
from google_stt.domain.models import (
    Role,
    Message,
    DialogueHistory,
    FillerSequence,
    PseudoVoicePitchMapper,
)

class TestMessage(unittest.TestCase):
    def test_message_initialization(self):
        """Messageが正しく初期化されること"""
        msg = Message(Role.USER, "こんにちは")
        self.assertEqual(msg.role, Role.USER)
        self.assertEqual(msg.text, "こんにちは")

    def test_to_gemini_format_user(self):
        """USERのMessageがGemini API用フォーマットに正しく変換されること"""
        msg = Message(Role.USER, "こんにちは")
        expected = {"role": "user", "parts": [{"text": "こんにちは"}]}
        self.assertEqual(msg.to_gemini_format(), expected)

    def test_to_gemini_format_model(self):
        """MODELのMessageがGemini API用フォーマットに正しく変換されること"""
        msg = Message(Role.MODEL, "返答です")
        expected = {"role": "model", "parts": [{"text": "返答です"}]}
        self.assertEqual(msg.to_gemini_format(), expected)

    def test_to_gemini_format_empty(self):
        """空文字列のMessageも正しく変換されること"""
        msg = Message(Role.USER, "")
        expected = {"role": "user", "parts": [{"text": ""}]}
        self.assertEqual(msg.to_gemini_format(), expected)


class TestDialogueHistory(unittest.TestCase):
    def test_history_initialization(self):
        """DialogueHistoryがシステムプロンプトと共に正しく初期化されること"""
        history = DialogueHistory(system_instruction="あなたはアシスタントです。")
        self.assertEqual(history.system_instruction, "あなたはアシスタントです。")
        self.assertEqual(len(history.get_messages()), 0)

    def test_add_user_message(self):
        """ユーザー発話が正常に履歴に追加され、トリムされること"""
        history = DialogueHistory()
        history.add_user_message("  テスト発話  ")
        messages = history.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, Role.USER)
        self.assertEqual(messages[0].text, "テスト発話")

    def test_add_model_message(self):
        """モデル発話が正常に履歴に追加され、トリムされること"""
        history = DialogueHistory()
        history.add_model_message("  モデルテスト  ")
        messages = history.get_messages()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, Role.MODEL)
        self.assertEqual(messages[0].text, "モデルテスト")

    def test_add_empty_message_ignored(self):
        """空文字列や空白のみの発話は追加を無視されること"""
        history = DialogueHistory()
        history.add_user_message("")
        history.add_user_message("   ")
        history.add_model_message("")
        history.add_model_message("   ")
        self.assertEqual(len(history.get_messages()), 0)

    def test_to_gemini_contents(self):
        """履歴全体がGemini用フォーマット配列に一括変換されること"""
        history = DialogueHistory()
        history.add_user_message("こんにちは")
        history.add_model_message("どうも")
        
        expected = [
            {"role": "user", "parts": [{"text": "こんにちは"}]},
            {"role": "model", "parts": [{"text": "どうも"}]},
        ]
        self.assertEqual(history.to_gemini_contents(), expected)

    def test_clear_history(self):
        """clear()で履歴がリセットされること"""
        history = DialogueHistory()
        history.add_user_message("こんにちは")
        self.assertEqual(len(history.get_messages()), 1)
        
        history.clear()
        self.assertEqual(len(history.get_messages()), 0)


class TestFillerSequence(unittest.TestCase):
    def test_filler_looping(self):
        """指定シーケンス順に文字が返され、末尾に達するとループすること"""
        seq = FillerSequence(["え", "ー", "と"])
        
        # 1周目
        self.assertEqual(seq.next_char(), "え")
        self.assertEqual(seq.next_char(), "ー")
        self.assertEqual(seq.next_char(), "と")
        
        # 2周目（ループ）
        self.assertEqual(seq.next_char(), "え")
        self.assertEqual(seq.next_char(), "ー")

    def test_filler_reset(self):
        """reset()でインデックスが初期位置に戻ること"""
        seq = FillerSequence(["え", "ー", "と"])
        self.assertEqual(seq.next_char(), "え")
        self.assertEqual(seq.next_char(), "ー")
        
        seq.reset()
        self.assertEqual(seq.next_char(), "え")

    def test_empty_sequence_raises_value_error(self):
        """空配列で初期化した場合はValueErrorが発生すること"""
        with self.assertRaises(ValueError):
            FillerSequence([])


class TestPseudoVoicePitchMapper(unittest.TestCase):
    def test_pitch_mapping(self):
        """文字のUnicode値に基づき、指定周波数リスト内で正しくマッピングされること"""
        freqs = [100.0, 200.0, 300.0]
        mapper = PseudoVoicePitchMapper(scale_frequencies=freqs, skip_characters=[" ", "\n"])
        
        # 'a' の ord は 97。 97 % 3 = 1。よって freqs[1] = 200.0
        self.assertEqual(mapper.map_char_to_frequency("a"), 200.0)
        
        # 'b' の ord は 98。 98 % 3 = 2。よって freqs[2] = 300.0
        self.assertEqual(mapper.map_char_to_frequency("b"), 300.0)

        # 全角文字 'あ' (ordは12354)。 12354 % 3 = 0。よって freqs[0] = 100.0
        self.assertEqual(mapper.map_char_to_frequency("あ"), 100.0)

    def test_skip_characters_return_none(self):
        """スキップ対象文字が渡された場合はNoneを返すこと"""
        freqs = [100.0, 200.0]
        mapper = PseudoVoicePitchMapper(scale_frequencies=freqs, skip_characters=[" ", "\n", "　"])
        
        self.assertIsNone(mapper.map_char_to_frequency(" "))
        self.assertIsNone(mapper.map_char_to_frequency("\n"))
        self.assertIsNone(mapper.map_char_to_frequency("　"))


if __name__ == "__main__":
    unittest.main()
