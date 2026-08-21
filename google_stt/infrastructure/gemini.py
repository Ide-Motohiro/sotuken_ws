import time
from typing import Any, Dict, Iterator, Optional
from google import genai
from google_stt.domain.interfaces import LanguageModel
from google_stt.domain.models import DialogueHistory

class GeminiLanguageModel(LanguageModel):
    """Vertex AI (Gemini 2.5 Flash API) を使った対話モデル具象実装"""
    def __init__(
        self,
        project_id: str = "gen-lang-client-0698198570",
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None
    ) -> None:
        """
        thinking_budget: Gemini 2.5 系の思考トークン数の上限。
            None なら API の既定（思考する）。0 で思考を無効化する。

            制約の多いシステムプロンプトほど思考が長引き、応答生成が遅くなる。実測では
            短い応答を求めるプロンプトで p50 3298ms、思考を切ると p50 579ms（約5.7倍）。
            疑似言語音では語彙が聞き手に届かないため内容の精度が効きにくい一方、
            レイテンシは研究の従属変数そのものなので、その用途では 0 が適する
            （詳細と留保は DECISIONS.md の「Gemini の thinking」参照）。素の音声合成を使う場合は内容が
            そのまま届くため、既定のままにしておくこと。
        """
        self.project_id: str = project_id
        self.location: str = location
        self.model_name: str = model_name
        self.system_instruction: Optional[str] = system_instruction
        self.thinking_budget: Optional[int] = thinking_budget
        #: 直近の生成にかかった秒数。一括は応答が返るまで、ストリーミングは
        #: 最初の断片が届くまで（体感に効くのはそこなので）。
        self.last_generation_sec: Optional[float] = None
        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def _build_config(self) -> Dict[str, Any]:
        """generate_content に渡す設定を組み立てる。指定の無い項目は入れない"""
        config: Dict[str, Any] = {}
        if self.system_instruction:
            config["system_instruction"] = self.system_instruction
        if self.thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": self.thinking_budget}
        return config

    def generate_reply(self, history: DialogueHistory) -> str:
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=history.to_gemini_contents(),
            config=self._build_config()
        )
        self.last_generation_sec = time.perf_counter() - started
        return response.text.strip() if response.text else ""

    def generate_reply_stream(self, history: DialogueHistory) -> Iterator[str]:
        started = time.perf_counter()
        self.last_generation_sec = None
        stream = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=history.to_gemini_contents(),
            config=self._build_config()
        )
        for chunk in stream:
            if chunk.text:
                # 最初の断片が届いた時点を記録する（そこから音を出し始められるため）
                if self.last_generation_sec is None:
                    self.last_generation_sec = time.perf_counter() - started
                for char in chunk.text:
                    yield char
