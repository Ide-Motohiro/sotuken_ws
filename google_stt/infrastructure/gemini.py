from typing import Iterator, Optional
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
        system_instruction: Optional[str] = None
    ) -> None:
        self.project_id: str = project_id
        self.location: str = location
        self.model_name: str = model_name
        self.system_instruction: Optional[str] = system_instruction
        self._client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )

    def generate_reply(self, history: DialogueHistory) -> str:
        config = {}
        if self.system_instruction:
            config["system_instruction"] = self.system_instruction
            
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=history.to_gemini_contents(),
            config=config
        )
        return response.text.strip() if response.text else ""

    def generate_reply_stream(self, history: DialogueHistory) -> Iterator[str]:
        config = {}
        if self.system_instruction:
            config["system_instruction"] = self.system_instruction

        stream = self._client.models.generate_content_stream(
            model=self.model_name,
            contents=history.to_gemini_contents(),
            config=config
        )
        for chunk in stream:
            if chunk.text:
                for char in chunk.text:
                    yield char
