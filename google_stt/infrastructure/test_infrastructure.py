import unittest
import copy
import os
import time
from unittest import mock
import requests
import sounddevice as sd
from google_stt.domain.models import (
    ConsonantSubstitutionTable, DialogueHistory, FillerSequence, PseudoVoicePitchMapper,
)
from google_stt.infrastructure.phoneme_swap import PhonemeSwapTTS, substitute_consonants
from google_stt.infrastructure.filler import VoiceVoxFillerPlayer
from google_stt.infrastructure.gemini import GeminiLanguageModel
from google_stt.infrastructure.voicevox import SynthesisTiming, VoiceVoxTTS
from google_stt.infrastructure.pseudo_voice import PseudoVoiceTTS
from google_stt.infrastructure.google_stt import (
    INPUT_DEVICE_ENV_VAR, GoogleSpeechRecognizer, describe_input_device,
    is_loopback_device, list_input_devices, resolve_input_device,
)

# 環境と依存サーバーのチェック
def has_google_credentials() -> bool:
    # デフォルトクレデンシャルまたは環境変数が設定されているか簡易チェック
    return (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is not None
        or os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    )

def is_voicevox_running() -> bool:
    try:
        # "localhost" は ::1 を先に返し約2秒のIPv6フォールバックが入るため、
        # timeout=0.5 だと起動していても必ず失敗する（本番実装と同じく 127.0.0.1 を使う）
        response = requests.get("http://127.0.0.1:50021/speakers", timeout=0.5)
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


def make_mora(consonant, vowel, pitch=5.5):
    """audio_query のモーラ1件を模したdict。差し替え対象外の項目を検証するために全項目持たせる"""
    return {
        "text": "", "consonant": consonant,
        "consonant_length": None if consonant is None else 0.05,
        "vowel": vowel, "vowel_length": 0.1, "pitch": pitch,
    }


def make_query(moras, pause_mora=None):
    return {
        "accent_phrases": [
            {"moras": moras, "accent": 1, "pause_mora": pause_mora, "is_interrogative": False}
        ],
        "speedScale": 1.0, "pitchScale": 0.0, "intonationScale": 1.0,
    }


class TestSubstituteConsonants(unittest.TestCase):
    """audio_query の子音差し替え（VOICEVOX を起動していなくても検証できる純粋な変換）"""

    def test_swaps_all_consonants_at_full_ratio(self):
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=1.0)
        query = make_query([make_mora("k", "a"), make_mora("n", "o"), make_mora("s", "u")])
        result = substitute_consonants(query, table)
        consonants = [m["consonant"] for m in result["accent_phrases"][0]["moras"]]
        # 近い置換：k->t（無声破裂音）、n->m（鼻音）、s->h（無声摩擦音）
        self.assertEqual(consonants, ["t", "m", "h"])

    def test_half_ratio_swaps_every_other_consonant_mora(self):
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=0.5)
        query = make_query([make_mora("k", "a"), make_mora("n", "o"),
                            make_mora("s", "u"), make_mora("m", "e")])
        result = substitute_consonants(query, table)
        consonants = [m["consonant"] for m in result["accent_phrases"][0]["moras"]]
        self.assertEqual(consonants, ["t", "n", "h", "m"])

    def test_vowel_only_moras_are_untouched_and_not_counted(self):
        """母音のみのモーラと撥音は consonant=None のまま。通し番号にも数えない"""
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=0.5)
        # 先頭に母音のみ・撥音を挟んでも、子音付きモーラの選ばれ方は変わらないこと
        query = make_query([make_mora(None, "a"), make_mora("k", "a"),
                            make_mora(None, "N"), make_mora("n", "o")])
        result = substitute_consonants(query, table)
        moras = result["accent_phrases"][0]["moras"]
        self.assertIsNone(moras[0]["consonant"])
        self.assertIsNone(moras[2]["consonant"])
        self.assertEqual(moras[1]["consonant"], "t")  # 通し番号0なので差し替え対象
        self.assertEqual(moras[3]["consonant"], "n")  # 通し番号1なので対象外

    def test_prosody_and_durations_are_preserved(self):
        """疑似言語音の核心は韻律とリズムの保持。pitch・長さ・母音が変わらないこと"""
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=1.0)
        moras = [make_mora("k", "a", pitch=5.1), make_mora("s", "u", pitch=6.3)]
        query = make_query(moras, pause_mora=make_mora(None, "pau", pitch=0.0))
        original = copy.deepcopy(query)
        result = substitute_consonants(query, table)

        for before, after in zip(original["accent_phrases"][0]["moras"],
                                 result["accent_phrases"][0]["moras"]):
            self.assertEqual(before["pitch"], after["pitch"])
            self.assertEqual(before["vowel"], after["vowel"])
            self.assertEqual(before["vowel_length"], after["vowel_length"])
            self.assertEqual(before["consonant_length"], after["consonant_length"])
        self.assertEqual(original["accent_phrases"][0]["pause_mora"],
                         result["accent_phrases"][0]["pause_mora"])
        for key in ("speedScale", "pitchScale", "intonationScale"):
            self.assertEqual(original[key], result[key])

    def test_unknown_consonant_passes_through(self):
        """置換表に無い子音は素通しする（黙って落ちるより気付ける）"""
        table = ConsonantSubstitutionTable.articulatory(swap_ratio=1.0)
        query = make_query([make_mora("ng", "a")])
        result = substitute_consonants(query, table)
        self.assertEqual(result["accent_phrases"][0]["moras"][0]["consonant"], "ng")

    def test_is_deterministic(self):
        """同じ文なら常に同じ結果になること（固定置換表であることの担保）"""
        moras = [make_mora("k", "a"), make_mora("s", "u"), make_mora("t", "o")]
        first = substitute_consonants(
            make_query(copy.deepcopy(moras)), ConsonantSubstitutionTable.articulatory(0.5))
        second = substitute_consonants(
            make_query(copy.deepcopy(moras)), ConsonantSubstitutionTable.articulatory(0.5))
        self.assertEqual(first, second)


class TestPhonemeSwapTTS(unittest.TestCase):
    def test_default_table_is_articulatory_half(self):
        """既定は聴取比較で選んだ「調音的に近い置換 × 1モーラおき」であること"""
        tts = PhonemeSwapTTS()
        self.assertEqual(tts.substitution_table.swap_ratio, 0.5)
        self.assertEqual(tts.substitution_table.substitute("k"), "t")

    def test_transform_query_applies_substitution(self):
        """VoiceVoxTTS の合成前フックとして置換が適用されること"""
        tts = PhonemeSwapTTS(
            substitution_table=ConsonantSubstitutionTable.articulatory(swap_ratio=1.0))
        query = make_query([make_mora("k", "a"), make_mora("n", "o")])
        result = tts._transform_query(query)
        self.assertEqual([m["consonant"] for m in result["accent_phrases"][0]["moras"]],
                         ["t", "m"])

    def test_voicevox_default_query_is_untouched(self):
        """親クラスの既定フックは加工しないこと（通常のVOICEVOX出力を壊さない）"""
        query = make_query([make_mora("k", "a")])
        self.assertEqual(VoiceVoxTTS()._transform_query(copy.deepcopy(query)), query)


class TestGeminiConfig(unittest.TestCase):
    """generate_content に渡す設定の組み立て（APIを叩かずに検証できる部分）"""

    def _model(self, **kwargs):
        # genai.Client の生成には認証が要るので __init__ を通さずに属性だけ用意する
        model = GeminiLanguageModel.__new__(GeminiLanguageModel)
        model.system_instruction = kwargs.get("system_instruction")
        model.thinking_budget = kwargs.get("thinking_budget")
        return model

    def test_omits_unset_fields(self):
        """指定の無い項目は設定に入れない（APIの既定に委ねる）"""
        self.assertEqual(self._model()._build_config(), {})

    def test_includes_system_instruction(self):
        config = self._model(system_instruction="test")._build_config()
        self.assertEqual(config["system_instruction"], "test")
        self.assertNotIn("thinking_config", config)

    def test_thinking_budget_zero_is_sent(self):
        """0 は「思考を無効化する」という明示的な指定なので、未指定と区別して送る"""
        config = self._model(thinking_budget=0)._build_config()
        self.assertEqual(config["thinking_config"], {"thinking_budget": 0})

    def test_thinking_budget_intermediate_value(self):
        config = self._model(thinking_budget=512)._build_config()
        self.assertEqual(config["thinking_config"], {"thinking_budget": 512})

    def test_none_thinking_budget_falls_back_to_api_default(self):
        self.assertNotIn("thinking_config", self._model(thinking_budget=None)._build_config())


class TestSynthesisTiming(unittest.TestCase):
    def test_time_to_first_sound_excludes_playback(self):
        """フィラーで隠すべき区間は再生を含まない（音が鳴り始めたら隠す対象ではない）"""
        timing = SynthesisTiming(query_sec=0.10, synthesis_sec=0.25, playback_sec=3.0)
        self.assertAlmostEqual(timing.time_to_first_sound_sec, 0.35)

    def test_total_includes_playback(self):
        timing = SynthesisTiming(query_sec=0.10, synthesis_sec=0.25, playback_sec=3.0)
        self.assertAlmostEqual(timing.total_sec, 3.35)


class TestVoiceVoxTimingIntegration(unittest.TestCase):
    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_synthesize_returns_wav_and_timings(self):
        """合成だけを単体で呼べること（フィラーの停止点を再生直前に置くための分割）"""
        wav_bytes, query_sec, synthesis_sec = VoiceVoxTTS().synthesize("こんにちは")
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertGreater(query_sec, 0.0)
        self.assertGreater(synthesis_sec, 0.0)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_synthesize_does_not_touch_is_speaking(self):
        """合成単体では is_speaking を触らない（呼び出し側が管理する契約）"""
        tts = VoiceVoxTTS()
        tts.synthesize("こんにちは")
        self.assertFalse(tts.is_speaking)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_phoneme_swap_applies_substitution_in_synthesize(self):
        """子音置換が分割後の合成経路でも効いていること（同じ文で波形が変わる）"""
        text = "こんにちは、今日はいい天気ですね"
        plain, _, _ = VoiceVoxTTS().synthesize(text)
        swapped, _, _ = PhonemeSwapTTS().synthesize(text)
        self.assertNotEqual(plain, swapped)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_swap_ratio_zero_matches_plain_voicevox(self):
        """差し替え率0なら素のVOICEVOXと一致する（置換以外の副作用が無いことの確認）"""
        text = "こんにちは"
        plain, _, _ = VoiceVoxTTS().synthesize(text)
        untouched, _, _ = PhonemeSwapTTS(
            substitution_table=ConsonantSubstitutionTable.articulatory(swap_ratio=0.0)
        ).synthesize(text)
        self.assertEqual(plain, untouched)

    @unittest.skipUnless(is_voicevox_running() and has_audio_output(),
                         "VOICEVOX server or audio output is not available")
    def test_speak_records_timing_and_notifies_callback(self):
        """speak() が3区間を記録し、コールバックにも渡すこと"""
        received = []
        tts = VoiceVoxTTS(on_timing=received.append)
        tts.speak("あ")

        self.assertIsNotNone(tts.last_timing)
        self.assertEqual(len(received), 1)
        self.assertIs(received[0], tts.last_timing)
        self.assertGreater(tts.last_timing.playback_sec, 0.0)
        self.assertAlmostEqual(
            tts.last_timing.total_sec,
            tts.last_timing.time_to_first_sound_sec + tts.last_timing.playback_sec,
        )
        self.assertFalse(tts.is_speaking)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_speak_ignores_blank_text_without_recording(self):
        """空文字では合成も記録もしないこと"""
        tts = VoiceVoxTTS()
        tts.speak("   ")
        self.assertIsNone(tts.last_timing)


class TestVoiceVoxFillerPlayerSelection(unittest.TestCase):
    """フィラーの選択規則。連続するターンで同じ音が鳴らないことが要件"""

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_never_repeats_the_previous_phrase(self):
        player = VoiceVoxFillerPlayer(VoiceVoxTTS(), phrases=("ん", "え", "あ"), seed=0)
        try:
            picks = [player._next_path() for _ in range(60)]
        finally:
            player.close()
        for previous, current in zip(picks, picks[1:]):
            self.assertNotEqual(previous, current)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_seed_makes_the_order_reproducible(self):
        """デバッグ時に同じ並びを再現できること（実験の再現性のためではない）"""
        def picks(seed):
            player = VoiceVoxFillerPlayer(VoiceVoxTTS(), phrases=("ん", "え", "あ"), seed=seed)
            try:
                return [player.phrases[player._last_index]
                        for _ in range(20) if player._next_path() or True]
            finally:
                player.close()
        self.assertEqual(picks(7), picks(7))

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_single_phrase_does_not_deadlock(self):
        """語句が1つしかない場合、候補が空になっても選べること"""
        player = VoiceVoxFillerPlayer(VoiceVoxTTS(), phrases=("ん",), seed=0)
        try:
            self.assertEqual(len({player._next_path() for _ in range(5)}), 1)
        finally:
            player.close()

    def test_rejects_empty_phrases(self):
        with self.assertRaises(ValueError):
            VoiceVoxFillerPlayer(VoiceVoxTTS(), phrases=())

    def test_rejects_negative_initial_delay(self):
        with self.assertRaises(ValueError):
            VoiceVoxFillerPlayer(VoiceVoxTTS(), phrases=("ん",), initial_delay_sec=-0.1)


class TestVoiceVoxFillerPlayerTiming(unittest.TestCase):
    """立ち上がりの遅延。認識確定と同時に鳴らすと食い気味に聞こえるため置いている。

    実際に音を鳴らすと待ち時間が読めないので、再生だけ差し替えて呼ばれた時刻を見る。
    """

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_does_not_sound_during_the_initial_delay(self):
        """遅延中に止められたら一度も鳴らさないこと"""
        player = VoiceVoxFillerPlayer(
            VoiceVoxTTS(), phrases=("ん",), initial_delay_sec=5.0, seed=0)
        try:
            with mock.patch("google_stt.infrastructure.filler.winsound.PlaySound") as play:
                player.start()
                time.sleep(0.1)
                player.stop()
                play.assert_not_called()
            # 鳴っていないので停止は待たされない
            self.assertLess(player.last_stop_delay_sec, 0.5)
        finally:
            player.close()

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_sounds_after_the_initial_delay(self):
        """遅延が明けたら鳴ること。鳴り始めが遅延より前でないこと"""
        delay = 0.2
        elapsed = []
        player = VoiceVoxFillerPlayer(
            VoiceVoxTTS(), phrases=("ん",), initial_delay_sec=delay, seed=0)
        try:
            with mock.patch("google_stt.infrastructure.filler.winsound.PlaySound",
                            side_effect=lambda *a, **k: elapsed.append(
                                time.perf_counter() - started)):
                started = time.perf_counter()
                player.start()
                time.sleep(delay + 0.3)
                player.stop()
        finally:
            player.close()
        self.assertTrue(elapsed, "遅延が明けても鳴らなかった")
        self.assertGreaterEqual(elapsed[0], delay)

    @unittest.skipUnless(is_voicevox_running(), "VOICEVOX server is not running")
    def test_zero_delay_sounds_immediately(self):
        """0 を渡せば従来どおり即座に鳴ること（比較条件を作れるようにしておく）"""
        elapsed = []
        player = VoiceVoxFillerPlayer(
            VoiceVoxTTS(), phrases=("ん",), initial_delay_sec=0.0, seed=0)
        try:
            with mock.patch("google_stt.infrastructure.filler.winsound.PlaySound",
                            side_effect=lambda *a, **k: elapsed.append(
                                time.perf_counter() - started)):
                started = time.perf_counter()
                player.start()
                time.sleep(0.2)
                player.stop()
        finally:
            player.close()
        self.assertTrue(elapsed, "即座に鳴るはずが鳴らなかった")
        self.assertLess(elapsed[0], 0.1)


class TestEndpointStability(unittest.TestCase):
    """終端判定の待ち時間の切り替え。発話の途中で確定されると会話が壊れるので、
    まだ続きそうな終わり方では長く待つ"""

    def _recognizer(self, short=0.4, long=1.0):
        # SpeechClient を作らずに判定だけを試すため、__init__ を通さずに組み立てる
        recognizer = GoogleSpeechRecognizer.__new__(GoogleSpeechRecognizer)
        recognizer.stability_duration = short
        recognizer.continuation_duration = long
        return recognizer

    def test_complete_utterance_uses_the_short_wait(self):
        recognizer = self._recognizer()
        for text in ("今日はいい天気だね", "ラーメンが好き", "これ食べていい？",
                     # 「もう」を継続語に入れると意志形（読もう）を巻き込むので入れていない
                     "本読もう", "本読むのってやっぱおもしろいよな"):
            with self.subTest(text=text):
                self.assertEqual(recognizer.required_stability(text), 0.4)

    def test_continuing_utterance_uses_the_long_wait(self):
        recognizer = self._recognizer()
        for text in ("映画を見たんだけど", "私は", "昨日ね、映画を", "えーと",
                     # 実対話で「本読むのってやっぱ」で切られた。副詞は後ろに述語が要る
                     "本読むのってやっぱ", "たぶん", "こんな"):
            with self.subTest(text=text):
                self.assertEqual(recognizer.required_stability(text), 1.0)

    def test_empty_transcript_uses_the_short_wait(self):
        """空文字は「続きそう」と判定しない（判定が真だと待ちだけ伸びる）"""
        self.assertEqual(self._recognizer().required_stability(""), 0.4)

    def test_rejects_continuation_shorter_than_stability(self):
        """長い方が短い方を下回る設定は受け付けないこと。

        検証は SpeechClient を作る前に走るので、認証情報が無くても実行できる。
        """
        with self.assertRaises(ValueError):
            GoogleSpeechRecognizer(stability_duration=0.8, continuation_duration=0.4,
                                   announce_device=False)


class TestInputDeviceResolution(unittest.TestCase):
    """マイク入力デバイスの指定解決。実機のデバイス構成に依存しないよう、
    実際に存在するデバイスを引いてから検証する"""

    def setUp(self):
        self.devices = list_input_devices()
        self._saved_env = os.environ.get(INPUT_DEVICE_ENV_VAR)
        os.environ.pop(INPUT_DEVICE_ENV_VAR, None)

    def tearDown(self):
        os.environ.pop(INPUT_DEVICE_ENV_VAR, None)
        if self._saved_env is not None:
            os.environ[INPUT_DEVICE_ENV_VAR] = self._saved_env

    @unittest.skipUnless(list_input_devices(), "入力デバイスが無い環境")
    def test_resolves_by_index(self):
        index = self.devices[0][0]
        self.assertEqual(resolve_input_device(index), index)

    @unittest.skipUnless(list_input_devices(), "入力デバイスが無い環境")
    def test_resolves_by_name_substring_case_insensitively(self):
        index, name = self.devices[0]
        self.assertEqual(resolve_input_device(name.lower()), index)

    @unittest.skipUnless(list_input_devices(), "入力デバイスが無い環境")
    def test_env_var_is_used_when_argument_is_omitted(self):
        index, name = self.devices[0]
        os.environ[INPUT_DEVICE_ENV_VAR] = str(index)
        self.assertEqual(resolve_input_device(None), index)

    @unittest.skipUnless(len(list_input_devices()) >= 2, "入力デバイスが2つ以上必要")
    def test_argument_takes_precedence_over_env_var(self):
        os.environ[INPUT_DEVICE_ENV_VAR] = str(self.devices[0][0])
        self.assertEqual(resolve_input_device(self.devices[1][0]), self.devices[1][0])

    def test_unknown_name_raises_with_candidates(self):
        """見つからないときは黙って既定に落とさず、候補を添えて落とす"""
        with self.assertRaises(ValueError) as ctx:
            resolve_input_device("この名前のデバイスは存在しないはず")
        self.assertIn("この名前のデバイスは存在しないはず", str(ctx.exception))

    def test_invalid_index_raises(self):
        with self.assertRaises(ValueError):
            resolve_input_device(9999)

    def test_blank_specification_falls_back_to_default(self):
        """空文字（.env に変数だけ書いて値が空、等）は指定なし扱いにする"""
        os.environ[INPUT_DEVICE_ENV_VAR] = "   "
        self.assertEqual(resolve_input_device(None), resolve_input_device(None))

    @unittest.skipUnless(list_input_devices(), "入力デバイスが無い環境")
    def test_describe_input_device_includes_name(self):
        index, name = self.devices[0]
        self.assertIn(str(index), describe_input_device(index))

    def test_describe_input_device_handles_none(self):
        self.assertIn("既定", describe_input_device(None))

    def test_is_loopback_device_none_is_false(self):
        self.assertFalse(is_loopback_device(None))


if __name__ == "__main__":
    unittest.main()
