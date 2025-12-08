from faster_whisper import WhisperModel, BatchedInferencePipeline
import os
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    


def speech_to_text(file):
    model_size = "turbo"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    segments, info = batched_model.transcribe(save_path, beam_size=16)
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    text = "".join(segment.text for segment in segments)
    return text




# model_size = "turbo" # "large-v3"

# # Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")
# batched_model = BatchedInferencePipeline(model=model)

# # or run on GPU with INT8
# # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# # or run on CPU with INT8
# # model = WhisperModel(model_size, device="cpu", compute_type="int8")

# segments, info = batched_model.transcribe("./uploads/1750836172331.mp3", batch_size=16)

# print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# text = "".join(segment.text for segment in segments)
# print(text)
# # segments = list(segments)

# # print(segments)

# # for segment in segments:
# #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))