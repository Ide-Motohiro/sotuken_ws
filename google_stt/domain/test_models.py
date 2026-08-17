import unittest
from google_stt.domain.models import (
    Role,
    Message,
    DialogueHistory,
    FillerSequence,
    PseudoVoicePitchMapper,
    ConsonantSubstitutionTable,
    ARTICULATION_GROUPS,
    CONSONANTS,
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


class TestConsonantSubstitutionTable(unittest.TestCase):
    def test_articulatory_table_covers_all_consonants(self):
        """調音グループ由来の置換表が全子音を覆い、置換先も既知の子音であること"""
        table = ConsonantSubstitutionTable.articulatory()
        self.assertEqual(set(table.mapping.keys()), set(CONSONANTS))
        self.assertTrue(set(table.mapping.values()).issubset(set(CONSONANTS)))

    def test_articulatory_table_stays_within_group(self):
        """置換先が同じ調音グループ内に留まること（調音様式・有声性が保たれる）"""
        table = ConsonantSubstitutionTable.articulatory()
        for group in ARTICULATION_GROUPS:
            for consonant in group:
                self.assertIn(table.substitute(consonant), group)

    def test_distant_table_covers_all_consonants(self):
        """比較対照用の遠い置換表も全子音を覆うこと"""
        table = ConsonantSubstitutionTable.distant()
        self.assertEqual(set(table.mapping.keys()), set(CONSONANTS))
        self.assertTrue(set(table.mapping.values()).issubset(set(CONSONANTS)))

    def test_no_fixed_points(self):
        """自分自身へ写る子音があると、そこだけ加工が効かず素のまま鳴ってしまう"""
        for table in (ConsonantSubstitutionTable.articulatory(),
                      ConsonantSubstitutionTable.distant()):
            for src, dst in table.mapping.items():
                self.assertNotEqual(src, dst)

    def test_rejects_fixed_point_mapping(self):
        """自分自身へ写る対応を渡したら生成時に弾くこと"""
        with self.assertRaises(ValueError):
            ConsonantSubstitutionTable({"k": "k"})

    def test_rejects_out_of_range_ratio(self):
        """swap_ratio が 0.0〜1.0 の外なら弾くこと"""
        with self.assertRaises(ValueError):
            ConsonantSubstitutionTable({"k": "t"}, swap_ratio=1.5)
        with self.assertRaises(ValueError):
            ConsonantSubstitutionTable({"k": "t"}, swap_ratio=-0.1)

    def test_substitute_returns_none_for_unknown(self):
        """表に無い子音は None（呼び出し側で素通しさせる）"""
        table = ConsonantSubstitutionTable.articulatory()
        self.assertIsNone(table.substitute("ng"))

    def test_should_swap_ratio_half_selects_every_other(self):
        """swap_ratio=0.5 は 0, 2, 4, ... 番目を選ぶ（聴取比較で採用した水準）"""
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=0.5)
        selected = [i for i in range(10) if table.should_swap(i)]
        self.assertEqual(selected, [0, 2, 4, 6, 8])

    def test_should_swap_ratio_one_selects_all(self):
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=1.0)
        self.assertTrue(all(table.should_swap(i) for i in range(10)))

    def test_should_swap_ratio_zero_selects_none(self):
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=0.0)
        self.assertFalse(any(table.should_swap(i) for i in range(10)))

    def test_should_swap_approximates_ratio(self):
        """中間の割合でも、選ばれる個数が指定した割合におおむね一致すること"""
        for ratio in (0.25, 0.33, 0.67, 0.75):
            table = ConsonantSubstitutionTable.articulatory(swap_ratio=ratio)
            count = sum(1 for i in range(100) if table.should_swap(i))
            self.assertAlmostEqual(count / 100, ratio, delta=0.02)

    def test_should_swap_rejects_negative_index(self):
        table = ConsonantSubstitutionTable.articulatory()
        with self.assertRaises(ValueError):
            table.should_swap(-1)


if __name__ == "__main__":
    unittest.main()
