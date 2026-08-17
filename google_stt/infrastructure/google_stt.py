import os
import queue
import threading
import time
from typing import List, Optional, Tuple, Union
import sounddevice as sd
from google.cloud import speech
from google_stt.domain.interfaces import SpeechRecognizer

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
        is_speaking_fn = lambda: False,
        device: Optional[Union[int, str]] = None,
        announce_device: bool = True,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.stability_duration: float = stability_duration
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

    def _callback(self, indata, _frames, _timestamp, _status):
        if not self.is_speaking_fn():
            self.audio_queue.put(bytes(indata))

    def recognize_once(self) -> str:
        """マイク入力を開始し、発話が確定（stability_durationの間変化なし）するまでブロックして文字列を返す"""
        # 前の発話で残った古いキューをクリア
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        stop_streaming = threading.Event()
        last_interim = ""
        stable_since = 0.0

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
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript
                    if result.is_final:
                        stop_streaming.set()
                        return transcript
                    
                    now = time.monotonic()
                    if transcript != last_interim:
                        last_interim = transcript
                        stable_since = now
                    elif transcript and (now - stable_since) >= self.stability_duration:
                        stop_streaming.set()
                        return transcript
                    print(f"\r途中: {transcript}", end="", flush=True)
        return ""
