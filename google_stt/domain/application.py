from typing import Callable, Optional
from google_stt.domain.interfaces import (
    SpeechRecognizer, LanguageModel, TextToSpeech, FillerPlayer
)
from google_stt.domain.models import DialogueHistory, TurnTiming

class DialogueApplicationService:
    """音声対話システムの1サイクルのユースケースを統制するアプリケーションサービス"""

    def __init__(
        self,
        recognizer: SpeechRecognizer,
        model: LanguageModel,
        tts: TextToSpeech,
        history: DialogueHistory,
        use_stream: bool = True,
        silence_marker: Optional[str] = None,
        max_idle_utterances: int = 1,
        on_turn: Optional[Callable[[str, str], None]] = None,
        filler: Optional[FillerPlayer] = None
    ) -> None:
        """
        silence_marker: 相手が黙ったまま時間切れになったときに、履歴へ入れる印。
            None なら何もせず次の認識へ戻る（従来どおり、常に相手の発話を待つ）。
            文字列を渡すと、それをユーザー発話として履歴に入れて応答を生成し、
            **こちらから話しかける**。認識器が `last_timed_out` を立てる実装でないと働かない。

        max_idle_utterances: 相手の発話を挟まずに続けて話しかける回数の上限。
            既定1。相手が席を外しているときに延々と喋り続けるのを防ぐ。

        filler: 応答生成の待機中にフィラーを鳴らす実装。None ならフィラー無し
            （これが実験条件「フィラーの有無」の切り替えになる）。

            **停止のタイミングは TTS 側のフックで決める。** `VoiceVoxTTS.on_before_playback`
            に `filler.stop` を配線しておくと、音声合成が終わって再生が始まる直前で止まる。
            配線しない場合は下の finally で止まるが、それでは応答の再生が終わるまで
            フィラーが鳴り続けて音が重なるため、必ず配線すること。

        on_turn: 1ターンが終わるたびに (認識テキスト, 応答テキスト) を受け取るコールバック。

            疑似言語音では応答が聞き手に語彙として届かないため、**認識結果を記録しておかないと
            会話が脱線した原因を後から追えない**。ASRが疑似言語音や雑音を拾うと、LLMが
            そこから意味をでっち上げて自然に見える応答を返すことがあり、出力を見ているだけでは
            異常に気付けない（DECISIONS.md の問題2）。実験時は必ず記録すること。
        """
        self.recognizer: SpeechRecognizer = recognizer
        self.model: LanguageModel = model
        self.tts: TextToSpeech = tts
        self.history: DialogueHistory = history
        self.use_stream: bool = use_stream
        self.on_turn: Optional[Callable[[str, str], None]] = on_turn
        self.filler: Optional[FillerPlayer] = filler
        self.silence_marker: Optional[str] = silence_marker
        self.max_idle_utterances: int = max_idle_utterances
        #: 相手の発話を挟まずに続けて話しかけた回数
        self._idle_utterances: int = 0
        #: 直近のターンの時間の内訳。各実装が測った値を束ねたもの。
        #: on_turn が呼ばれる時点では既に入っているので、ログ側はここを読めばよい。
        self.last_turn_timing: Optional[TurnTiming] = None

    def run_once(self) -> None:
        """
        1サイクルの音声対話ユースケースを実行する。
        1. 音声認識からテキストを取得する
        2. 空文字でなければ、履歴にユーザー発話を追加する
        3. 設定（use_stream）に基づき、一括またはストリーミングで応答を生成・再生する
        4. 生成された応答を履歴にモデル発話として追加する
        5. on_turn が設定されていれば、認識テキストと応答テキストを渡す
        """
        user_text = self.recognizer.recognize_once()
        stripped_user_text = user_text.strip() if user_text else ""

        if not stripped_user_text:
            if self._should_speak_up():
                self._speak_up()
            return

        self._idle_utterances = 0

        self.history.add_user_message(stripped_user_text)

        # 応答生成の前に開始する。生成を待ってから始めたのでは隠す対象が過ぎている
        if self.filler is not None:
            self.filler.start()
        try:
            reply_text = self._generate_and_speak()
        finally:
            # 通常は TTS のフックが既に止めている。ここは配線漏れ・例外時の保険（stop は冪等）
            if self.filler is not None:
                self.filler.stop()

        self.last_turn_timing = self._collect_timing()

        if self.on_turn is not None:
            self.on_turn(stripped_user_text, reply_text)

    def _should_speak_up(self) -> bool:
        """相手が黙ったままなので、こちらから話しかけるべきかを判断する"""
        return (self.silence_marker is not None
                and self.recognizer.last_timed_out
                and self._idle_utterances < self.max_idle_utterances)

    def _speak_up(self) -> None:
        """相手が黙ったままのとき、こちらから話しかける。

        **フィラーは鳴らさない。** フィラーは「相手が喋り終わってから応答が出るまで」を
        埋めるものなので、相手の発話が無いこの経路で鳴らすと、フィラーの回数が
        相手の黙り方に左右されることになり、実験条件（フィラーの有無）の交絡になる。
        """
        self._idle_utterances += 1
        self.history.add_user_message(self.silence_marker)
        reply_text = self._generate_and_speak()
        self.last_turn_timing = self._collect_timing()
        if self.on_turn is not None:
            self.on_turn(self.silence_marker, reply_text)

    def _collect_timing(self) -> TurnTiming:
        """各実装が測った区間を1つに束ねる。測っていない実装では None が入る。

        ここで測り直さないのが要点。区間の境目を知っているのはそれぞれの実装であり、
        アプリケーション層から測ると内訳ではなく合計しか取れない。
        """
        return TurnTiming(
            endpoint_wait_sec=self.recognizer.last_endpoint_wait_sec,
            generation_sec=self.model.last_generation_sec,
            time_to_first_sound_sec=self.tts.last_time_to_first_sound_sec,
            playback_sec=self.tts.last_playback_sec,
            filler_stop_delay_sec=(
                getattr(self.filler, "last_stop_delay_sec", None)
                if self.filler is not None else None),
        )

    def _generate_and_speak(self) -> str:
        """応答を生成して再生し、履歴に追加したテキストを返す"""
        if self.use_stream:
            # ストリーミング応答の開始
            stream = self.model.generate_reply_stream(self.history)
            
            # 再生した内容を横取りして蓄積するスパイジェネレータ
            accumulated_chars = []
            def spy_generator():
                for char in stream:
                    accumulated_chars.append(char)
                    yield char

            self.tts.speak_stream(spy_generator())

            # 蓄積した文字列を結合して履歴に追加
            reply_text = "".join(accumulated_chars)
        else:
            # 一括応答の生成と再生
            reply_text = self.model.generate_reply(self.history)
            self.tts.speak(reply_text)

        self.history.add_model_message(reply_text)
        return reply_text
