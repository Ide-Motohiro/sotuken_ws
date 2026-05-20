from RealtimeSTT import AudioToTextRecorder


def on_text(text: str) -> None:
    text = text.strip()
    if text:
        print(f"認識結果: {text}")


if __name__ == "__main__":
    print("準備完了。話しかけてください。（Ctrl+C で終了）\n")
    recorder = AudioToTextRecorder(language="ja", model="small")
    while True:
        recorder.text(on_text)
