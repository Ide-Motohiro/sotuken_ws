"""疑似音声の音質を聞き比べるための一時スクリプト（unittest discoverには含めない）。

4段階で比較する：
  1. 現状版：純音サイン波
  2. 改良案：倍音+ピッチグライド+微小ノイズ+揺らぎ（電子音を脱した「ビープ」）
  3. フォルマント版（文字ハッシュ）：母音をハッシュで割り当てるだけの簡易版
  4. フォルマント版（実読み）：pykakasiで実際のひらがな読みに変換し、
     かなごとの正しい母音リズムでフォルマント合成する。聞き取りやすさ重視。
"""
import time
import numpy as np
import sounddevice as sd
import pykakasi

from google_stt.domain.models import PseudoVoicePitchMapper

SAMPLE_RATE = 44100
VOLUME = 0.3
CHAR_DURATION = 0.08

PITCH_MAPPER = PseudoVoicePitchMapper(
    scale_frequencies=[261.6, 329.6, 392.0, 523.3, 659.3, 784.0, 1046.5],
    skip_characters=[" ", "　", "\n", "\r", "\t"]
)

SAMPLE_TEXT = "こんにちは、今日はいい天気ですね。何かお手伝いできることはありますか？"

# 日本語5母音の近似フォルマント周波数（F1, F2, F3, Hz）
VOWEL_FORMANTS = {
    "a": {"f1": 800, "f2": 1300, "f3": 2500},
    "i": {"f1": 300, "f2": 2300, "f3": 3000},
    "u": {"f1": 350, "f2": 1200, "f3": 2400},
    "e": {"f1": 500, "f2": 2000, "f3": 2800},
    "o": {"f1": 500, "f2": 900,  "f3": 2500},
}
FORMANT_BANDWIDTHS = (80, 110, 140)
FORMANT_WEIGHTS = (1.0, 0.55, 0.25)
FORMANT_SCALE = 1.15  # 声道を小さく（高め）にして親しみやすい印象にする

# 五十音（拗音・濁音・半濁音含む）のかな1文字 → 末尾母音
_VOWEL_GROUPS = {
    "a": "あかさたなはまやらわがざだばぱぁゃ",
    "i": "いきしちにひみりぎじぢびぴぃ",
    "u": "うくすつぬふむゆるぐずづぶぷぅゅ",
    "e": "えけせてねへめれげぜでべぺぇ",
    "o": "おこそとのほもよろごぞどぼぽぉょ",
}
KANA_TO_VOWEL = {kana: vowel for vowel, kana_set in _VOWEL_GROUPS.items() for kana in kana_set}
# 純粋な母音（あいうえお、小書き含む）＝子音の立ち上がりなし
PURE_VOWEL_KANA = set("あいうえおぁぃぅぇぉ")

_kakasi = pykakasi.kakasi()


def to_hiragana_reading(text: str) -> str:
    """漢字混じりの文章を実際の発音に基づいたひらがな読みに変換する"""
    return "".join(item["hira"] for item in _kakasi.convert(text))


def make_beep_current(freq: float, duration: float) -> np.ndarray:
    """現状実装：純音サイン波"""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    env = np.ones(n)
    attack = int(n * 0.1)
    release = int(n * 0.3)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return (np.sin(2 * np.pi * freq * t) * VOLUME * env).astype(np.float32)


def make_syllable_animalese(freq: float, duration: float, rng: np.random.Generator) -> np.ndarray:
    """改良案：どうぶつの森風の擬似音節
    - 基本周波数に微小なランダムジッターを加える（機械的な均一さを崩す）
    - 子音的な立ち上がりを模した下降ピッチグライド（高め→指定周波数）
    - 倍音を1つ重ねてサイン波特有の電子音っぽさを抑える
    - わずかなノイズで息っぽさ（フォルマント感）を加える
    """
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    jitter = rng.uniform(0.95, 1.05)
    f0 = freq * jitter

    # ピッチグライド：開始時は+20%高く、すぐにf0へ落ち着く（子音→母音の遷移を模す）
    freq_env = f0 * 1.2 * np.exp(-t / (duration * 0.15)) + f0 * (1 - np.exp(-t / (duration * 0.15)))
    phase = 2 * np.pi * np.cumsum(freq_env) / SAMPLE_RATE

    tone = np.sin(phase) * 0.75 + np.sin(2 * phase) * 0.2 + np.sin(3 * phase) * 0.05

    noise = rng.normal(0, 1, n)
    kernel = np.ones(5) / 5
    noise = np.convolve(noise, kernel, mode="same") * 0.06

    env = np.ones(n)
    attack = max(1, int(n * 0.15))
    release = max(1, int(n * 0.35))
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)

    signal = (tone + noise) * VOLUME * env
    return signal.astype(np.float32)


def _resonator_filter(x: np.ndarray, freq: float, bandwidth: float, sr: int) -> np.ndarray:
    """2次共振器（デジタルレゾネータ）。フォルマント帯域だけを共振させる簡易バンドパス。"""
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2 * np.pi * freq / sr
    a1 = 2 * r * np.cos(theta)
    a2 = -r * r
    gain = (1 - r ** 2) * np.sin(theta)
    if abs(gain) < 1e-6:
        gain = 1 - r ** 2

    y = np.empty_like(x)
    y1 = 0.0
    y2 = 0.0
    for i in range(len(x)):
        yi = gain * x[i] + a1 * y1 + a2 * y2
        y[i] = yi
        y2 = y1
        y1 = yi
    return y


def _make_glottal_source(f0_env: np.ndarray, n: int, rng: np.random.Generator, n_harmonics: int = 12) -> np.ndarray:
    """声帯振動を模した倍音列（1/kで減衰）+ わずかな息ノイズ"""
    phase0 = 2 * np.pi * np.cumsum(f0_env) / SAMPLE_RATE
    source = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        source += np.sin(k * phase0) / k

    noise = rng.normal(0, 1, n)
    kernel = np.ones(5) / 5
    noise = np.convolve(noise, kernel, mode="same")

    return source / n_harmonics + noise * 0.05


def _synthesize_vowel(vowel_key: str, has_onset: bool, duration: float, f0_base: float, rng: np.random.Generator) -> np.ndarray:
    vowel = VOWEL_FORMANTS[vowel_key]

    onset = np.zeros(0, dtype=np.float32)
    onset_dur = 0.0
    if has_onset:
        onset_dur = min(duration * 0.25, 0.02)
        n_onset = int(SAMPLE_RATE * onset_dur)
        burst = rng.normal(0, 1, n_onset) * VOLUME * 0.5
        onset = (burst * np.linspace(1, 0, n_onset)).astype(np.float32)

    n = int(SAMPLE_RATE * (duration - onset_dur))
    f0_env = np.linspace(f0_base * 0.97, f0_base * 1.05, n)  # 語尾でわずかに上がる（明るい印象）

    source = _make_glottal_source(f0_env, n, rng)

    formant_out = np.zeros(n)
    for f, bw, w in zip((vowel["f1"], vowel["f2"], vowel["f3"]), FORMANT_BANDWIDTHS, FORMANT_WEIGHTS):
        formant_out += _resonator_filter(source, f * FORMANT_SCALE, bw, SAMPLE_RATE) * w

    peak = np.max(np.abs(formant_out)) + 1e-9
    formant_out /= peak

    env = np.ones(n)
    attack = max(1, int(n * 0.1))
    release = max(1, int(n * 0.3))
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)

    vowel_signal = (formant_out * VOLUME * env).astype(np.float32)
    return np.concatenate([onset, vowel_signal])


def make_syllable_formant(char: str, duration: float, rng: np.random.Generator) -> np.ndarray:
    """フォルマント版（文字ハッシュ）：母音を文字コードのハッシュで割り当てるだけの簡易版"""
    if char in PITCH_MAPPER.skip_characters:
        return None

    vowel_key = list(VOWEL_FORMANTS.keys())[ord(char) % len(VOWEL_FORMANTS)]
    f0_base = (300 + (ord(char) % 8) * 15) * rng.uniform(0.95, 1.05)
    has_onset = (ord(char) % 3) != 0
    return _synthesize_vowel(vowel_key, has_onset, duration, f0_base, rng)


def make_syllable_formant_kana(kana: str, duration: float, rng: np.random.Generator) -> np.ndarray:
    """フォルマント版（実読み）：かな1文字の実際の母音に基づいて合成する

    あ行（純粋な母音）は子音の立ち上がりなし、それ以外のかなは子音的なノイズ
    バーストを前置する。「ん」は鼻音的な低い共鳴のみ、「っ」は無音（促音の間）として扱う。
    """
    if kana in PITCH_MAPPER.skip_characters or kana in "、。！？「」『』,.!?":
        return None

    f0_base = (300 + (ord(kana) % 8) * 15) * rng.uniform(0.95, 1.05)

    if kana == "っ":
        return None  # 促音：短い無音の間
    if kana == "ん":
        # 鼻音：フォルマントを使わず低い共鳴の鼻音的な響きのみ
        n = int(SAMPLE_RATE * duration)
        f0_env = np.linspace(f0_base * 0.9, f0_base * 0.85, n)
        source = _make_glottal_source(f0_env, n, rng)
        nasal = _resonator_filter(source, 250, 100, SAMPLE_RATE)
        peak = np.max(np.abs(nasal)) + 1e-9
        nasal /= peak
        env = np.ones(n)
        attack = max(1, int(n * 0.2))
        release = max(1, int(n * 0.4))
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        return (nasal * VOLUME * 0.7 * env).astype(np.float32)
    if kana == "ー":
        return None  # 長音：簡易版では未対応（直前の母音を伸ばす処理は今後の課題）

    vowel_key = KANA_TO_VOWEL.get(kana)
    if vowel_key is None:
        return None

    has_onset = kana not in PURE_VOWEL_KANA
    return _synthesize_vowel(vowel_key, has_onset, duration, f0_base, rng)


def play_text_freq_based(text: str, synth_fn, rng=None):
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as out_stream:
        for c in text:
            freq = PITCH_MAPPER.map_char_to_frequency(c)
            if freq is None:
                time.sleep(CHAR_DURATION)
                continue
            beep = synth_fn(freq, CHAR_DURATION, rng) if rng is not None else synth_fn(freq, CHAR_DURATION)
            out_stream.write(beep)


def play_text_char_based(text: str, synth_fn, rng: np.random.Generator):
    with sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as out_stream:
        for c in text:
            signal = synth_fn(c, CHAR_DURATION, rng)
            if signal is None or len(signal) == 0:
                time.sleep(CHAR_DURATION)
                continue
            out_stream.write(signal)


def main():
    rng = np.random.default_rng()

    print("【1】現状版（純音サイン波）を再生します...")
    play_text_freq_based(SAMPLE_TEXT, make_beep_current)
    time.sleep(0.8)

    print("【2】改良案（どうぶつの森風・倍音+グライド+ノイズ）を再生します...")
    play_text_freq_based(SAMPLE_TEXT, make_syllable_animalese, rng)
    time.sleep(0.8)

    print("【3】フォルマント版・文字ハッシュ（母音共振シミュレーション）を再生します...")
    play_text_char_based(SAMPLE_TEXT, make_syllable_formant, rng)
    time.sleep(0.8)

    hiragana_text = to_hiragana_reading(SAMPLE_TEXT)
    print(f"【4】フォルマント版・実読み（ひらがな変換: {hiragana_text}）を再生します...")
    play_text_char_based(hiragana_text, make_syllable_formant_kana, rng)

    print("完了。")


if __name__ == "__main__":
    main()
