from google_stt.domain.interfaces import SpeechRecognizer, LanguageModel, TextToSpeech
from google_stt.domain.models import DialogueHistory

class DialogueApplicationService:
    """音声対話システムの1サイクルのユースケースを統制するアプリケーションサービス"""
    
    def __init__(
        self,
        recognizer: SpeechRecognizer,
        model: LanguageModel,
        tts: TextToSpeech,
        history: DialogueHistory,
        use_stream: bool = True
    ) -> None:
        self.recognizer: SpeechRecognizer = recognizer
        self.model: LanguageModel = model
        self.tts: TextToSpeech = tts
        self.history: DialogueHistory = history
        self.use_stream: bool = use_stream

    def run_once(self) -> None:
        """
        1サイクルの音声対話ユースケースを実行する。
        1. 音声認識からテキストを取得する
        2. 空文字でなければ、履歴にユーザー発話を追加する
        3. 設定（use_stream）に基づき、一括またはストリーミングで応答を生成・再生する
        4. 生成された応答を履歴にモデル発話として追加する
        """
        user_text = self.recognizer.recognize_once()
        stripped_user_text = user_text.strip() if user_text else ""
        
        if not stripped_user_text:
            return

        self.history.add_user_message(stripped_user_text)

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
            self.history.add_model_message(reply_text)
        else:
            # 一括応答の生成と再生
            reply = self.model.generate_reply(self.history)
            self.tts.speak(reply)
            self.history.add_model_message(reply)
