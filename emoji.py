emotions = {"happy": "😊", "sad": "😕", "angry": "😡"}

message=input("Enter your message: ")

for word, emoji in emotions.items():
    message=message.replace(word,emoji)

print("Converted message:", message)
