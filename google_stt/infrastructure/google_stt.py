import array
import os
import queue
import statistics
import threading
import time
from typing import Callable, List, Optional, Tuple, Union
import sounddevice as sd
from google.cloud import speech
from google_stt.domain.interfaces import SpeechRecognizer
from google_stt.domain.models import looks_incomplete

#: 入力デバイスを指定する環境変数。`.env` に書けばコードもコマンドも触らずに切り替えられる。
#: 値はデバイス番号（"7"）でも名前の一部（"Microphone Array"）でもよい。
INPUT_DEVICE_ENV_VAR = "STT_INPUT_DEVICE"

#: ループバック（システム自身の再生音を拾う）デバイスの名前に現れる語。
#: これが既定になっているとマイクの声が入らず、認識が永久に返ってこないので警告する。
LOOPBACK_KEYWORDS = ("ステレオ ミキサー", "ステレオミキサー", "stereo mix", "what u hear")


def list_input_devices() -> List[Tuple[int, str]]:
    """入力チャンネルを持つデバイスの (番号, 名前) を列挙する"""
    return [
        (i, device["name"])
        for i, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


def measure_input_level(
    index: Optional[int], seconds: float = 5.0, sample_rate: int = 16000,
    chunk_size: int = 1600, on_tick: Optional[Callable[[float], None]] = None,
) -> Tuple[int, float, float]:
    """指定デバイスから一定時間録音し、(ブロック数, RMSの中央値, RMSの最大) を返す。

    マイクの選択を間違えると recognize_once() が例外も出さずに返ってこないだけなので、
    対話を回す前にここで切り分ける。音は保存しない（音量だけ見る）。

    on_tick は残り秒数を毎秒渡す。カウントダウンの表示に使う。
    """
    levels: List[float] = []

    def callback(indata, _frames, _timestamp, _status):
        samples = array.array("h")
        samples.frombytes(bytes(indata))
        if samples:
            levels.append((sum(float(v) * v for v in samples) / len(samples)) ** 0.5)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16",
                        blocksize=chunk_size, callback=callback, device=index):
        remaining = seconds
        while remaining > 0:
            if on_tick is not None:
                on_tick(remaining)
            step = min(1.0, remaining)
            time.sleep(step)
            remaining -= step

    if not levels:
        return 0, 0.0, 0.0
    return len(levels), statistics.median(levels), max(levels)


def resolve_input_device(device: Optional[Union[int, str]]) -> Optional[int]:
    """入力デバイスの指定を番号へ解決する。

    優先順位は 引数 > 環境変数 STT_INPUT_DEVICE > sounddevice の既定デバイス。
    名前で指定された場合は大文字小文字を無視した部分一致で探し、複数該当したら先頭を採る
    （同じデバイスがホストAPIごとに重複して並ぶため）。

    明示的に指定されたものが見つからない場合は、候補を添えて例外を投げる。指定が無く
    既定デバイスも取れない場合だけ None を返し、sounddevice の判断に委ねる。
    """
    requested = device if device is not None else os.environ.get(INPUT_DEVICE_ENV_VAR)
    if requested is None or (isinstance(requested, str) and not requested.strip()):
        default = sd.default.device[0] if sd.default.device is not None else None
        return default if isinstance(default, int) and default >= 0 else None

    candidates = list_input_devices()

    if isinstance(requested, int) or (isinstance(requested, str) and requested.strip().lstrip("-").isdigit()):
        index = int(requested)
        if not any(i == index for i, _ in candidates):
            raise ValueError(
                f"入力デバイス番号 {index} は入力チャンネルを持ちません。\n"
                f"  利用できる入力デバイス:\n" + _format_devices(candidates)
            )
        return index

    keyword = str(requested).strip().lower()
    matched = [(i, name) for i, name in candidates if keyword in name.lower()]
    if not matched:
        raise ValueError(
            f"入力デバイス名に「{requested}」を含むものが見つかりません。\n"
            f"  利用できる入力デバイス:\n" + _format_devices(candidates)
        )
    return matched[0][0]


def _format_devices(devices: List[Tuple[int, str]]) -> str:
    return "\n".join(f"    {i:>3}  {name}" for i, name in devices)


def describe_input_device(index: Optional[int]) -> str:
    """表示用にデバイス名を返す。番号が None のときは sounddevice の既定を指す旨を返す"""
    if index is None:
        return "(sounddevice の既定デバイス)"
    try:
        return f"{index}: {sd.query_devices(index)['name']}"
    except Exception as e:
        return f"{index}: (デバイス情報を取得できません: {type(e).__name__}: {e})"


def is_loopback_device(index: Optional[int]) -> bool:
    """指定デバイスがループバック（システムの再生音を拾う）らしいかを名前から判定する"""
    if index is None:
        return False
    try:
        name = sd.query_devices(index)["name"].lower()
    except Exception:
        return False
    return any(keyword.lower() in name for keyword in LOOPBACK_KEYWORDS)


class GoogleSpeechRecognizer(SpeechRecognizer):
    """Google Cloud Speech-to-Text (Streaming API) を使った音声認識具象実装"""
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,
        stability_duration: float = 0.4,
        continuation_duration: float = 1.0,
        silence_timeout_sec: Optional[float] = None,
        push_to_talk: bool = False,
        is_speaking_fn = lambda: False,
        device: Optional[Union[int, str]] = None,
        announce_device: bool = True,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        #: 中間結果がこの秒数変化しなければ発話終端とみなす。
        #: **ここはフィラーが鳴り始める前の区間**なので、伸ばすと隠せない無音がそのまま増える。
        self.stability_duration: float = stability_duration
        #: まだ続きそうな終わり方（助詞・接続助詞・言いよどみ）のときに使う長い方の待ち。
        #: 日本語の発話中の休止は文節末に集中するため、そこだけ待てば
        #: テンポを保ったまま途中で切られるのを減らせる。
        if continuation_duration < stability_duration:
            raise ValueError(
                f"continuation_duration ({continuation_duration}) は "
                f"stability_duration ({stability_duration}) 以上にすること")
        self.continuation_duration: float = continuation_duration
        #: 何も認識しないままこの秒数が過ぎたら空文字を返して制御を戻す。
        #: 相手が黙ったままのときにこちらから話しかけられるようにするための口。
        #: None または 0以下 なら時間切れ無し（発話があるまで待ち続ける）。
        self.silence_timeout_sec: Optional[float] = silence_timeout_sec
        #: True なら、キーを押している間だけマイクを開く（手押しトリガー）。
        #: 終端はキーを離した時刻そのものになるので、stability_duration も
        #: continuation_duration も使われない。周りの声を拾う経路も閉じる。
        self.push_to_talk: bool = push_to_talk
        #: 押している間だけ録音するキー。pynput のキー定数を差し替えれば変えられる。
        #: 将来ぬいぐるみ側の物理ボタンに置き換えるなら、この2つを差し替える。
        self.push_to_talk_key = None
        self.push_to_talk_label: str = "スペース"
        if push_to_talk:
            from pynput import keyboard
            self.push_to_talk_key = keyboard.Key.space
        self.is_speaking_fn = is_speaking_fn
        self.audio_queue: queue.Queue = queue.Queue()

        # 実験のたびにOSの既定設定へ依存すると条件間で入力経路が変わりうるので、
        # 実際に使うデバイスを起動時に解決して明示する
        self.device: Optional[int] = resolve_input_device(device)
        if announce_device:
            print(f"入力デバイス: {describe_input_device(self.device)}")
            if is_loopback_device(self.device):
                print(
                    "  ⚠ これはループバック（システム自身の再生音を拾う）デバイスです。\n"
                    "    マイクの声が入らず認識が返ってこない可能性があります。\n"
                    f"    .env に {INPUT_DEVICE_ENV_VAR}=Microphone のように書くか、\n"
                    "    --input-device で指定してください（--list-devices で一覧）。"
                )

        self._stt_client = speech.SpeechClient()
        self._streaming_config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code="ja-JP",
            ),
            interim_results=True,
        )

    def required_stability(self, transcript: str) -> float:
        """この中間結果を確定させるまでに必要な無変化時間を返す。

        まだ続きそうな終わり方なら長い方を使う。判定を誤って長い方に倒れても、
        損は待ち時間だけで発話が壊れることはない。
        """
        return self.continuation_duration if looks_incomplete(transcript) else self.stability_duration

    def _callback(self, indata, _frames, _timestamp, _status):
        if not self.is_speaking_fn():
            self.audio_queue.put(bytes(indata))

    def recognize_once(self) -> str:
        """マイク入力を開始し、発話が確定するまでブロックして文字列を返す。

        確定条件は「中間結果が一定時間変化しないこと」。必要な時間は終わり方で変わり、
        まだ続きそうなら continuation_duration、そうでなければ stability_duration。

        push_to_talk が真のときは、キーを押している間だけ録音して離した時点で確定する。
        """
        if self.push_to_talk:
            return self._recognize_while_held()

        # 前の発話で残った古いキューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        stop_streaming = threading.Event()
        last_interim = ""
        stable_since = 0.0
        # 今回の発話ぶんの計測値。取れなかった経路では None のまま残る
        self.last_endpoint_wait_sec = None
        self.last_confidence = None
        self.last_finalized_by_service = None
        self.last_timed_out = False

        # 何も認識しないまま時間切れになったら、送信側を止めてストリームを畳む。
        # 一度でも認識結果が届いたら取り消す（喋っている途中で切らないため）。
        timed_out = threading.Event()
        timeout_timer: Optional[threading.Timer] = None
        if self.silence_timeout_sec and self.silence_timeout_sec > 0:
            def on_timeout() -> None:
                timed_out.set()
                stop_streaming.set()
            timeout_timer = threading.Timer(self.silence_timeout_sec, on_timeout)
            timeout_timer.daemon = True
            timeout_timer.start()

        def generate_requests():
            while not stop_streaming.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        # マイク入力を開始して Google STT にストリーミング送信
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                            blocksize=self.chunk_size, callback=self._callback,
                            device=self.device):
            responses = self._stt_client.streaming_recognize(self._streaming_config, generate_requests())
            try:
                for response in responses:
                    for result in response.results:
                        alternative = result.alternatives[0]
                        transcript = alternative.transcript
                        now = time.monotonic()
                        if transcript and timeout_timer is not None:
                            timeout_timer.cancel()   # 喋り始めたので時間切れは取り消す
                            timeout_timer = None
                        if result.is_final:
                            # 認識サービス側が確定と判断した経路。信頼度はここにしか入らない
                            self._record_endpoint(now, stable_since, alternative, by_service=True)
                            stop_streaming.set()
                            return transcript

                        if transcript != last_interim:
                            last_interim = transcript
                            stable_since = now
                        elif transcript and (now - stable_since) >= self.required_stability(transcript):
                            # 中間結果が変化しなくなった経路。本システムでは主にこちらが効く
                            self._record_endpoint(now, stable_since, alternative, by_service=False)
                            stop_streaming.set()
                            return transcript
                        print(f"\r途中: {transcript}", end="", flush=True)
            finally:
                if timeout_timer is not None:
                    timeout_timer.cancel()
        # ここに来るのは、何も認識しないままストリームが終わったとき。
        # 時間切れによるものかどうかを残す（相手が黙っているのかの判断に使う）
        self.last_timed_out = timed_out.is_set()
        return ""

    @staticmethod
    def assemble_transcript(finals, interim: str) -> str:
        """確定した断片と、確定していない途中結果をつないで1つの発話にする。

        押しっぱなしの間に認識サービスが複数回確定させることがあるため、
        最後の1つだけを採ると前半が落ちる。
        """
        return "".join(finals) + interim

    def _recognize_while_held(self) -> str:
        """キーを押している間だけ録音し、離したら確定して返す（手押しトリガー）。

        終端の判定を人が行うので、中間結果の無変化を待つ必要が無い。
        待ち時間が消えるだけでなく、**押していない間はマイクを開かないので
        周りの声を拾わない**。騒がしい場所での実演を想定している。
        """
        from pynput import keyboard

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.last_endpoint_wait_sec = None
        self.last_confidence = None
        self.last_finalized_by_service = None
        self.last_timed_out = False

        stop_streaming = threading.Event()
        pressed = threading.Event()
        released_at = []

        def on_press(key):
            if key == self.push_to_talk_key:
                pressed.set()

        def on_release(key):
            if key == self.push_to_talk_key and pressed.is_set():
                released_at.append(time.monotonic())
                stop_streaming.set()
                return False   # 離したら監視を終える

        def generate_requests():
            while not stop_streaming.is_set():
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()
        finals = []
        interim = ""
        try:
            print(f"  [{self.push_to_talk_label}を押しながら話す]", end="", flush=True)
            pressed.wait()          # 押されるまでマイクを開かない
            print("\r  [録音中... 離すと確定]      ", end="", flush=True)
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="int16",
                                blocksize=self.chunk_size, callback=self._callback,
                                device=self.device):
                responses = self._stt_client.streaming_recognize(
                    self._streaming_config, generate_requests())
                for response in responses:
                    for result in response.results:
                        alternative = result.alternatives[0]
                        if result.is_final:
                            finals.append(alternative.transcript)
                            interim = ""
                            self.last_confidence = getattr(alternative, "confidence", None)
                        else:
                            interim = alternative.transcript
                        print(f"\r  途中: {self.assemble_transcript(finals, interim)}",
                              end="", flush=True)
            # 終端はキーを離した時刻そのもの。ここに残すのは「離してから認識結果が
            # 出そろうまで」で、中間結果の無変化を待つ経路の待ち時間に相当する。
            if released_at:
                self.last_endpoint_wait_sec = max(0.0, time.monotonic() - released_at[0])
        finally:
            listener.stop()
            print()
        return self.assemble_transcript(finals, interim).strip()


    def _record_endpoint(self, now: float, stable_since: float, alternative,
                         by_service: bool) -> None:
        """発話終端を検出したときの計測値を残す。

        `last_endpoint_wait_sec` の起点は「中間結果が変化しなくなった時刻」であって、
        ユーザーが口を閉じた時刻ではない。中間結果は音声から遅れて届くため、
        実際の無音はこの値より長い（その遅れは未計測）。

        信頼度は Google STT が最終結果にしか入れないと考えられるため、
        interim 経路では 0.0 のまま返ることを想定している。判定できるよう
        生の値をそのまま残す（`by_service` でどちらの経路だったかが分かる）。
        """
        self.last_endpoint_wait_sec = (now - stable_since) if stable_since else None
        self.last_confidence = getattr(alternative, "confidence", None)
        self.last_finalized_by_service = by_service
