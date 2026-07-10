import dialogue

print("準備完了。テキストを入力してください。（Ctrl+C で終了）\n")

while True:
    try:
        user_text = input("あなた: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user_text:
        continue
    dialogue.chat_with_pseudo_voice(user_text)
