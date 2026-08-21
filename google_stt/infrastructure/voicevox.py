import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import requests
import winsound
from google_stt.domain.interfaces import TextToSpeech


@dataclass(frozen=True)
class SynthesisTiming:
    """1回の発話にかかった時間の内訳（秒）。

    区間名が VOICEVOX の API（audio_query / synthesis）に対応しているため、
    domain ではなく infrastructure に置いている。

    体感リアルタイム性がこの研究の主題なので、本番の経路の中で測れることが要る。
    オフラインでの計測は measure_voicevox_latency.py が別途行っている。
    """
    query_sec: float
    synthesis_sec: float
    playback_sec: float

    @property
    def time_to_first_sound_sec(self) -> float:
        """TTSが呼ばれてから最初の音が鳴るまで。
        フィラーで隠すべき無音区間のうち、TTSが占める分に相当する。"""
        return self.query_sec + self.synthesis_sec

    @property
    def total_sec(self) -> float:
        return self.query_sec + self.synthesis_sec + self.playback_sec


class VoiceVoxTTS(TextToSpeech):
    """VOICEVOX サーバーを用いた音声合成・再生の具象実装"""

    # "localhost" はこの環境では ::1（IPv6）を先に返し、VOICEVOX が IPv6 で待ち受けて
    # いないため、TCP接続のたびに約2秒のフォールバック待ちが入る
    # （measure_voicevox_latency.py で実測）。体感レイテンシに直接効くので 127.0.0.1 を既定にする。
    def __init__(
        self,
        url: str = "http://127.0.0.1:50021",
        speaker_id: int = 3,
        on_timing: Optional[Callable[[SynthesisTiming], None]] = None,
        on_before_playback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.url: str = url
        self.speaker_id: int = speaker_id
        self.is_speaking: bool = False
        #: 直近の発話の時間内訳。実験中のログ取得用
        self.last_timing: Optional[SynthesisTiming] = None
        #: 発話ごとに時間内訳を受け取るコールバック（省略可）
        self.on_timing: Optional[Callable[[SynthesisTiming], None]] = on_timing
        #: 合成が終わり再生が始まる直前に呼ばれるコールバック（省略可）。
        #: 待機中のフィラーを止める位置がここ。応答テキストが返った時点で止めると
        #: 合成の約1.1秒がまるごと無音になる。
        self.on_before_playback: Optional[Callable[[], None]] = on_before_playback

    def _transform_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """合成前に audio_query を加工する拡張点。既定では何もしない。

        疑似言語音のように「合成前のクエリを書き換える」派生実装のために用意している。
        合成後の波形を加工するのと違い、合成するのは VOICEVOX 自身なので音質が落ちない。
        """
        return query

    def synthesize(self, text: str) -> Tuple[bytes, float, float]:
        """テキストから wav バイト列を作る。音は鳴らさない。

        戻り値は (wavバイト列, audio_queryにかかった秒, synthesisにかかった秒)。
        再生と分けてあるのは、体感レイテンシの測定と、待機中のフィラーを
        「再生が始まる直前」で止めるため（フィラー機構は DECISIONS.md の「フィラー挿入機構」）。

        `is_speaking` はここでは触らない。単体で呼ぶ場合は呼び出し側が管理すること。
        """
        started = time.perf_counter()
        query_response = requests.post(
            f"{self.url}/audio_query",
            params={"text": text, "speaker": self.speaker_id},
        )
        query_response.raise_for_status()
        query = self._transform_query(query_response.json())
        query_sec = time.perf_counter() - started

        started = time.perf_counter()
        synthesis_response = requests.post(
            f"{self.url}/synthesis",
            params={"speaker": self.speaker_id},
            json=query,
        )
        # 失敗応答をそのまま wav として書くと再生時に不可解な失敗になるため、ここで落とす
        synthesis_response.raise_for_status()
        synthesis_sec = time.perf_counter() - started

        return synthesis_response.content, query_sec, synthesis_sec

    def play(self, wav_bytes: bytes) -> float:
        """wav バイト列を再生し、かかった秒数を返す。再生完了までブロックする。

        `is_speaking` はここでは触らない。単体で呼ぶ場合は呼び出し側が管理すること。
        """
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp = f.name
        try:
            winsound.PlaySound(tmp, winsound.SND_FILENAME)
        finally:
            os.unlink(tmp)
        return time.perf_counter() - started

    def _record_timing(self, timing: SynthesisTiming) -> None:
        self.last_timing = timing
        # インターフェース側の共通項目にも写す（アプリケーション層はこちらを見る）
        self.last_time_to_first_sound_sec = timing.time_to_first_sound_sec
        self.last_playback_sec = timing.playback_sec
        if self.on_timing is not None:
            self.on_timing(timing)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        # 合成中もマイクを止めたままにする（この間に発話されても取りこぼす仕様は従来通り）
        self.is_speaking = True
        try:
            wav_bytes, query_sec, synthesis_sec = self.synthesize(text)
            if self.on_before_playback is not None:
                self.on_before_playback()
            playback_sec = self.play(wav_bytes)
        finally:
            self.is_speaking = False
        self._record_timing(SynthesisTiming(query_sec, synthesis_sec, playback_sec))

    def speak_stream(self, text_stream: Iterator[str]) -> None:
        """ストリームテキストを一括結合してVOICEVOXで再生する"""
        full_text = "".join(text_stream)
        self.speak(full_text)
