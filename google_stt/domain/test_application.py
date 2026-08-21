import unittest
from google_stt.domain.models import DialogueHistory, Role
from google_stt.domain.mocks import (
    MockSpeechRecognizer,
    MockLanguageModel,
    MockTextToSpeech,
    MockFillerPlayer,
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


class TestDialogueApplicationServiceTurnLogging(unittest.TestCase):
    """on_turn による1ターンの記録。実験時に会話の脱線を後から追うために要る"""

    def _run(self, texts, reply, use_stream):
        recorded = []
        service = DialogueApplicationService(
            recognizer=MockSpeechRecognizer(texts),
            model=MockLanguageModel(reply=reply),
            tts=MockTextToSpeech(),
            history=DialogueHistory(),
            use_stream=use_stream,
            on_turn=lambda user, model_reply: recorded.append((user, model_reply)),
        )
        service.run_once()
        return recorded

    def test_reports_user_and_reply_in_batch_mode(self):
        self.assertEqual(self._run(["こんにちは"], "うん、いい、天気、だね", False),
                         [("こんにちは", "うん、いい、天気、だね")])

    def test_reports_user_and_reply_in_stream_mode(self):
        """ストリーミング時もスパイジェネレータで蓄積した全文が渡ること"""
        self.assertEqual(self._run(["こんにちは"], "うん、いい、天気、だね", True),
                         [("こんにちは", "うん、いい、天気、だね")])

    def test_reports_garbled_recognition_verbatim(self):
        """崩れた認識結果こそ記録の対象。LLMが意味をでっち上げても入力が残る"""
        recorded = self._run(["とんにつぃわ ぱんとお もい"], "おお、パン、牛乳、だね", False)
        self.assertEqual(recorded[0][0], "とんにつぃわ ぱんとお もい")

    def test_not_called_when_recognition_is_empty(self):
        """認識が空のターンは対話が起きていないので通知しない"""
        self.assertEqual(self._run([""], "応答しません", False), [])

    def test_omitted_callback_is_optional(self):
        """on_turn を渡さなくても従来通り動くこと"""
        tts = MockTextToSpeech()
        service = DialogueApplicationService(
            recognizer=MockSpeechRecognizer(["こんにちは"]),
            model=MockLanguageModel(reply="うん"),
            tts=tts,
            history=DialogueHistory(),
            use_stream=False,
        )
        service.run_once()
        self.assertEqual(tts.spoken_texts, ["うん"])



class RecordingTextToSpeech(MockTextToSpeech):
    """合成と再生を分けて記録するTTS。フィラーの停止位置を検証するために使う"""
    def __init__(self, events, filler=None):
        super().__init__()
        self.events = events
        self.filler = filler

    def speak(self, text: str) -> None:
        self.events.append("synthesize")
        # 本番の VoiceVoxTTS は合成後・再生前に on_before_playback を呼ぶ
        if self.filler is not None:
            self.filler.stop()
        self.events.append("play")
        self.spoken_texts.append(text)


class TestDialogueApplicationServiceTiming(unittest.TestCase):
    """区間ごとの計測値を1ターンぶんに束ねること。

    区間の境目を知っているのは各実装なので、アプリケーション層では測り直さず集めるだけ。
    測っていない実装では None のまま残ること（欠測と0秒を混同しない）も要件。
    """

    def _service(self, filler=None):
        recognizer = MockSpeechRecognizer(["こんにちは"])
        model = MockLanguageModel(reply="うん")
        tts = MockTextToSpeech()
        return DialogueApplicationService(
            recognizer=recognizer, model=model, tts=tts,
            history=DialogueHistory(), use_stream=False, filler=filler,
        ), recognizer, model, tts

    def test_collects_each_measured_section(self):
        service, recognizer, model, tts = self._service()
        recognizer.last_endpoint_wait_sec = 0.4
        model.last_generation_sec = 0.5
        tts.last_time_to_first_sound_sec = 0.7
        tts.last_playback_sec = 2.0

        service.run_once()

        timing = service.last_turn_timing
        self.assertEqual(timing.endpoint_wait_sec, 0.4)
        self.assertEqual(timing.generation_sec, 0.5)
        self.assertEqual(timing.time_to_first_sound_sec, 0.7)
        self.assertEqual(timing.playback_sec, 2.0)
        self.assertAlmostEqual(timing.time_to_response_sec, 1.6)

    def test_unmeasured_sections_stay_none(self):
        """計測しない実装では None のまま。0秒として扱わないこと"""
        service, _, _, _ = self._service()
        service.run_once()
        timing = service.last_turn_timing
        self.assertIsNone(timing.endpoint_wait_sec)
        self.assertIsNone(timing.time_to_response_sec)

    def test_filler_delay_is_added_to_the_response_delay(self):
        """フィラーを鳴らし終わるまで待った分も応答までの時間に含めること"""
        filler = MockFillerPlayer()
        filler.last_stop_delay_sec = 0.3
        service, recognizer, model, tts = self._service(filler=filler)
        recognizer.last_endpoint_wait_sec = 0.4
        model.last_generation_sec = 0.5
        tts.last_time_to_first_sound_sec = 0.7

        service.run_once()

        self.assertEqual(service.last_turn_timing.filler_stop_delay_sec, 0.3)
        self.assertAlmostEqual(service.last_turn_timing.time_to_response_sec, 1.9)

    def test_no_filler_leaves_the_delay_none(self):
        """フィラー無し条件では null。0秒（待たなかった）と区別する"""
        service, _, _, _ = self._service()
        service.run_once()
        self.assertIsNone(service.last_turn_timing.filler_stop_delay_sec)

    def test_timing_is_available_when_the_callback_runs(self):
        """on_turn の時点で計測値が入っていること（ログ側がここを読むため）"""
        seen = []
        service, recognizer, model, tts = self._service()
        recognizer.last_endpoint_wait_sec = 0.4
        model.last_generation_sec = 0.5
        tts.last_time_to_first_sound_sec = 0.7
        service.on_turn = lambda user, reply: seen.append(
            service.last_turn_timing.time_to_response_sec)

        service.run_once()

        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], 1.6)


class TestDialogueApplicationServiceFiller(unittest.TestCase):
    """フィラーの開始・停止のタイミング。これが実験の独立変数そのものなので順序が要"""

    def _service(self, filler, tts, texts=("こんにちは",)):
        return DialogueApplicationService(
            recognizer=MockSpeechRecognizer(list(texts)),
            model=MockLanguageModel(reply="うん、いい、天気、だね"),
            tts=tts,
            history=DialogueHistory(),
            use_stream=False,
            filler=filler,
        )

    def test_starts_before_generation_and_stops_before_playback(self):
        """停止は合成の後・再生の前。応答テキストが返った時点で止めると合成中が無音になる"""
        events = []
        filler = MockFillerPlayer()
        # フィラーの start/stop も同じ列に混ぜて順序を見る
        filler.events = events
        tts = RecordingTextToSpeech(events, filler=filler)
        self._service(filler, tts).run_once()

        self.assertEqual(events, ["start", "synthesize", "stop", "play"])

    def test_stopped_even_if_generation_fails(self):
        """例外時もフィラーが鳴りっぱなしにならないこと"""
        filler = MockFillerPlayer()

        class FailingModel(MockLanguageModel):
            def generate_reply(self, history):
                raise RuntimeError("生成に失敗")

        service = DialogueApplicationService(
            recognizer=MockSpeechRecognizer(["こんにちは"]),
            model=FailingModel(),
            tts=MockTextToSpeech(),
            history=DialogueHistory(),
            use_stream=False,
            filler=filler,
        )
        with self.assertRaises(RuntimeError):
            service.run_once()
        self.assertEqual(filler.events, ["start", "stop"])

    def test_not_started_when_recognition_is_empty(self):
        """発話が無ければ待機も無いので鳴らさない"""
        filler = MockFillerPlayer()
        self._service(filler, MockTextToSpeech(), texts=[""]).run_once()
        self.assertEqual(filler.events, [])

    def test_runs_without_filler(self):
        """フィラー無し条件（filler=None）でも通常どおり動くこと"""
        tts = MockTextToSpeech()
        self._service(None, tts).run_once()
        self.assertEqual(tts.spoken_texts, ["うん、いい、天気、だね"])

    def test_stop_is_idempotent(self):
        """TTSのフックと finally の両方から呼ばれるため冪等である必要がある"""
        filler = MockFillerPlayer()
        filler.start()
        filler.stop()
        filler.stop()
        self.assertEqual(filler.events, ["start", "stop"])

if __name__ == "__main__":
    unittest.main()
