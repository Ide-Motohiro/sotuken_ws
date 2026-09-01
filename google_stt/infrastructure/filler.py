"""応答生成の待機中に鳴らすフィラー（言い淀み）音の実装。

フィラーの有無は本研究の独立変数そのもの（sub-RQ1）なので、**静的で単純な方式**にしてある。
生成内容やタイミングを動的に変えると、条件間に「フィラーの有無」以外の差が入り込む。

設計の要点：

  - **起動時に事前合成する。** 鳴らす直前に合成すると、隠したいはずの遅延を自分で作ることになる。
    合成済みの wav は一時ファイルに置き、再生時はファイル読み出しだけで済ませる。
  - **すぐには鳴らさない。** 認識が確定した瞬間に鳴らすと食い気味に聞こえる（実際に対話して
    確認した）。人は相手が黙った瞬間に言い淀むのではなく、少し置いてから言い淀む。
    既定は0.4秒で、これは耳で決めた値。
    遅延中に停止されたら一度も鳴らさずに終わる。応答が異常に速く返ったときに
    フィラーだけ後から鳴る、という事故を防ぐため。
  - **フィラーを優先し、応答を待たせる。** 応答生成が速く終わっても、鳴っているフィラーは
    最後まで鳴らしてから応答へ移る。**応答を最速で届けることに価値を置かない**という判断。
    立ち上がり0.4秒＋フィラー最長0.92秒＝1.32秒に対し、最短の待機区間は1.04秒
    （Gemini 0.40 ＋ 合成 0.64）なので、速いターンでは最大0.28秒ほど応答が遅れる。
    そのぶんフィラーと応答が途切れずに繋がるので、音としてはむしろ自然になる。
    **フィラーあり条件だけ応答が遅くなるのは実験の交絡なので、隠さず測る。**
    待たされた秒数は `last_stop_delay_sec` に入り、`main_phoneme_swap.py --log` が
    `filler_stop_delay_sec` としてターンごとに記録する。
  - **ループしない。** 埋めるべき区間は約2.4秒、フィラー1つは約0.8秒なので、1つ鳴らすと
    残り1.5秒ほどが沈黙になる。これは人間の言い淀み（「えーと……」と言って少し黙る）として
    自然な形で、音で埋め尽くす方がむしろ不自然になる。
  - **応答生成が長引いたときだけ2つ目を鳴らす。** 間隔の既定は2.0秒で、これは実測に基づく。
    1つ目（約0.8秒）を言い終えてから2.0秒後＝発話終端から約2.8秒の時点で2つ目が出る。
    応答の音が鳴り始めるのは通常約2.0秒後なので、**普通のターンでは2つ目は出ない**。
    生成が長引いたときだけ出る、という当初の意図どおりになる。
    間隔が短すぎる（1.5秒など）と、2つ目を鳴らし始めた直後に応答ができてしまい、
    鳴り終わるまで待つぶん応答が約0.9秒遅れる（実測）。フィラーあり条件だけ応答が遅くなると
    実験の交絡になるので、ここは詰めておく価値がある。
    間隔は固定値にすること（動的に判断すると条件間で挙動がぶれる）。
  - **直前と同じものは選ばない無作為選択。** 決定的なローテーションだと周期に気付かれうる
    （5種なら5ターンで一巡する）。機械的に感じられると親しみやすさに逆行する。
    参加者ごとに並びを揃える意味は無い（ターン数も発話内容も人によって違うので、
    決定的に回しても実際には揃わない）。バグ追跡のために seed だけ渡せるようにしてある。

停止のタイミングは `VoiceVoxTTS.on_before_playback` に `stop` を配線して制御する。
応答テキストが返った時点で止めると、そのあとの音声合成 約1.1秒がまるごと無音になる。
"""
import os
import random
import tempfile
import threading
import time
from typing import List, Optional, Sequence

import winsound

from google_stt.domain.interfaces import FillerPlayer
from google_stt.infrastructure.voicevox import VoiceVoxTTS

#: 既定のフィラー。
#:
#: **子音置換は通さない**（`synthesizer` に素の VoiceVoxTTS を渡す）。
#: 「感動詞は子音を持たないので壊れない」と考えていたが、これが成り立つのは
#: 「うーん」「あー」だけだった。「えーと」の `と`、「あのー」の `の` は子音を持つ。
#: しかも語ごとに別々に合成するため、その子音は必ず通し番号0番になり、
#: `should_swap(0)` が常に真を返す。**差し替え率によらず必ず変換されていた**。
#:
#: 選択は無作為だが直前と同じものは選ばないので、連続するターンで同じ音は鳴らない
#: （選択状態はターンをまたいで保持される）。
DEFAULT_FILLER_PHRASES = ("えーと", "あのー", "うーん", "えーっと", "あー")


class VoiceVoxFillerPlayer(FillerPlayer):
    """VOICEVOX で事前合成したフィラーを、待機中に別スレッドで鳴らす実装。

    再生中は TTS と同じ `is_speaking` を立てる。立てないと自分のフィラーをマイクが拾い、
    音声認識がそれを文字に起こして対話が脱線する。
    """

    def __init__(
        self,
        tts: VoiceVoxTTS,
        phrases: Sequence[str] = DEFAULT_FILLER_PHRASES,
        synthesizer: Optional[VoiceVoxTTS] = None,
        repeat_interval_sec: float = 2.0,
        initial_delay_sec: float = 0.4,
        seed: Optional[int] = None,
    ) -> None:
        if not phrases:
            raise ValueError("フィラーの語句が空です")
        if initial_delay_sec < 0.0:
            raise ValueError(f"立ち上がりの遅延が負です: {initial_delay_sec}")
        self.tts: VoiceVoxTTS = tts
        #: 事前合成にだけ使う TTS。再生中の is_speaking は tts 側に立てる。
        #: **疑似言語音の子音置換を避けるために分けてある。** 省略すると tts をそのまま使う。
        self.synthesizer: VoiceVoxTTS = synthesizer if synthesizer is not None else tts
        self.phrases: List[str] = list(phrases)
        self.repeat_interval_sec: float = repeat_interval_sec
        #: start() から1つ目を鳴らし始めるまでの待ち。0 にすると認識確定と同時に鳴る
        self.initial_delay_sec: float = initial_delay_sec
        # seed=None なら実行ごとに並びが変わる。再現が要るときだけ値を渡す
        self._random = random.Random(seed)
        self._last_index: Optional[int] = None
        self._stop_event: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        #: 直近の stop() が返るまでにかかった秒数。再生中の1音を待つぶんの遅延が乗る
        self.last_stop_delay_sec: float = 0.0

        # 起動時に全パターンを合成して一時ファイル化する（再生時に合成しないため）
        self._paths: List[str] = []
        for phrase in self.phrases:
            wav_bytes, _, _ = self.synthesizer.synthesize(phrase)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav_bytes)
                self._paths.append(f.name)

    def _next_path(self) -> str:
        """直前と同じものを避けて次の音を無作為に選ぶ。選んだ位置は次回まで保持する"""
        candidates = [i for i in range(len(self._paths)) if i != self._last_index]
        if not candidates:  # 語句が1つしかない場合の保険
            candidates = list(range(len(self._paths)))
        index = self._random.choice(candidates)
        self._last_index = index
        return self._paths[index]

    def _run(self, stop_event: threading.Event) -> None:
        # 少し置いてから鳴らし始める。ここで止められたら一度も鳴らさない
        if stop_event.wait(self.initial_delay_sec):
            return
        # 1つ鳴らし、応答が来なければ固定間隔で次を鳴らす。連続では鳴らさない
        while not stop_event.is_set():
            winsound.PlaySound(self._next_path(), winsound.SND_FILENAME)
            if stop_event.wait(self.repeat_interval_sec):
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self.tts.is_speaking = True
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(self._stop_event,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """再生を止める。冪等。再生中の1音は最後まで鳴らしてから止まる。

        鳴っている途中で切ると語の途中で音が途切れて不自然になるため、待って止める。
        待った分だけ応答が遅れるが、それは承知のうえで**フィラーを優先する**（上の設計メモ参照）。
        待った秒数は last_stop_delay_sec に残るので、実験時はここを記録すること。
        まだ鳴り始めていなければ即座に返る。
        """
        if self._thread is None:
            return
        started = time.perf_counter()
        self._stop_event.set()
        self._thread.join()
        self.last_stop_delay_sec = time.perf_counter() - started
        self._thread = None
        self._stop_event = None
        self.tts.is_speaking = False

    def close(self) -> None:
        """一時ファイルを片付ける"""
        self.stop()
        for path in self._paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._paths = []
