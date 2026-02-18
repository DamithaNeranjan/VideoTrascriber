# transcriber.py
from faster_whisper import WhisperModel
import ffmpeg
import sys

VIDEO_PATH = sys.argv[1]
AUDIO_PATH = "audio.wav"
OUT_PATH = "transcription.txt"

# Extract audio
ffmpeg.input(VIDEO_PATH).output(
    AUDIO_PATH, ac=1, ar=16000
).overwrite_output().run(quiet=True)

# Load model (FAST + STABLE combo)
model = WhisperModel(
    "base",
    device="cuda",
    compute_type="float16"
)

segments, info = model.transcribe(
    AUDIO_PATH,
    beam_size=1,
    vad_filter=True,
    temperature=0.0
)

print(f"Detected language: {info.language}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(seg.text + "\n")

print("Transcription completed !!!")
