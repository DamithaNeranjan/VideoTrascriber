# main.py
import subprocess
import sys

VIDEO = "C:\\Projects\\VideoTranscriber\\EnglishNews.mp4"
# VIDEO = "C:\\Projects\\VideoTranscriber\\SinhalaNews.mp4"

result = subprocess.run(
    [sys.executable, "transcriber.py", VIDEO],
    capture_output=True,
    text=True
)

print(result.stdout)

if result.returncode != 0:
    print("Transcriber exited abnormally (expected on Windows + CUDA)")














# from datetime import datetime
#
# import ffmpeg
# from faster_whisper import WhisperModel
#
# print("Start Time:", datetime.now().time())
#
# # VIDEO_PATH = "C:\\Projects\\VideoTranscriber\\EnglishNews.mp4"
# VIDEO_PATH = "C:\\Projects\\VideoTranscriber\\SinhalaNews.mp4"
# AUDIO_PATH = "audio.wav"
#
# # Extract audio
# ffmpeg.input(VIDEO_PATH).output(
#     AUDIO_PATH, ac=1, ar="16000"
# ).overwrite_output().run()
#
# # Load model (GPU)
# model = WhisperModel(
#     "base",
#     device="cuda",
#     compute_type="float32"  # important for RTX GPUs
# )
# # model = WhisperModel(
# #     "base",
# #     device="cpu",
# #     compute_type="int8"
# # )
#
# # Transcribe
# segments, info = model.transcribe(AUDIO_PATH, language="si", task="transcribe")
#
# # Save text
# with open("transcription.txt", "w", encoding="utf-8") as f:
#     for segment in segments:
#         f.write(segment.text + " ")
#
# print("End Time:", datetime.now().time())
# print("Detected language:", info.language)
# print("Transcription completed ✅")
#
# del segments
# del info
# del model
#
# import gc
#
# gc.collect()
