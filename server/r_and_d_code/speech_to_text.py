
import whisper
model = whisper.load_model("medium") 
result = model.transcribe("./datasets/liridon_eminem-lose-yourself.mp3")
print(result["text"])